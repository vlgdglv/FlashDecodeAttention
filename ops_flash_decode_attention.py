import os
import numpy as np
import time
import math
from mindspore import ops
from mindspore.nn import Cell
from mindspore.ops import CustomRegOp, DataType
from mindnlp.core import ops as core_ops 

import mindspore as ms
from mindspore import context, Tensor
from mindspore.nn import Cell
from mindspore.ops import DataType, CustomRegOp
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype

from mindnlp.core import nn

context.set_context(
    # mode=ms.GRAPH_MODE,
    device_target="Ascend",
    save_graphs=False,
    save_graphs_path="./ms_graphs"
    )


class FlashDecodeAttentionNet(Cell):
    def __init__(self, func, o_shape):
        super(FlashDecodeAttentionNet, self).__init__()

        reg_info = (
            CustomRegOp("FlashDecodeAttention")
            # ---- inputs ----
            .input(0, "query_states", "required")
            .input(1, "key_states", "required")
            .input(2, "value_states", "required")
            .input(3, "o_weights", "required")
            # ---- outputs ----
            .output(0, "attn_output", "required")
            .dtype_format(
                ("bfloat16", "ND"),  # query_states
                ("bfloat16", "ND"),  # key_states
                ("bfloat16", "ND"),  # value_states
                ("bfloat16", "ND"),  # o_weights
                ("bfloat16", "ND"),  # attn_output
            )
            .target("Ascend")
            .get_op_info()
        )

        def out_shape_fn(*input_shapes):
            return (o_shape)
        
        def out_dtype_fn(*input_dtypes):
            return (mstype.bfloat16)

        self.flash_decode_attention = ops.Custom(
            func=func,
            out_shape=out_shape_fn,
            out_dtype=out_dtype_fn,
            func_type="aot",
            bprop=None,
            reg_info=reg_info
        )

    def construct(self,
                  query_states,
                  key_states,
                  value_states,
                  o_weights,
                  ):
        # print(">>> before fused_qkv")
        attn_output = self.flash_decode_attention(
            query_states, key_states, value_states, o_weights
            )
        # print(">>> after fused_qkv")
        return attn_output

from mindspore import Profiler
from mindspore.profiler import ProfilerLevel, ProfilerActivity, AicoreMetrics
    
profiler = Profiler(
    output_path=os.path.join("/home/ma-user/work/FlashDecodeAttention/profiler_out", "eager"),
    profiler_level=ProfilerLevel.Level1,
    activities=[ProfilerActivity.NPU],
    aic_metrics=AicoreMetrics.PipeUtilization,
    start_profile=False,
)

def eager_calcs(query_states, key_states, value_states, o_weights):
    query_states = query_states.to(ms.float32)
    key_states = key_states.to(ms.float32)
    value_states = value_states.to(ms.float32)
    
    bsz, q_len, num_heads, head_dim = 1, 1, 32, 128
    attn_weights = core_ops.matmul(query_states, core_ops.transpose(key_states, 2, 3)) / math.sqrt(head_dim)
            
   
    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=ms.float32).to(query_states.dtype)

    # attn_output: [BS(1), HEAD_NUM(32), SEQ_LEN, HEAD_DIM(128)]    
    # [BS, HEAD_NUM, Q_SEQ, HEAD_DIM]
    attn_output = core_ops.matmul(attn_weights, value_states)

    #[BS, Q_SEQ, HEAD_NUM, HEAD_DIM]]
    attn_output = core_ops.transpose(attn_output, 1, 2)
    
    #[BS, Q_SEQ, DIM(HEAD_NUM*HEAD_DIM)]
    attn_output = attn_output.reshape(bsz, q_len, -1)
    return attn_output

if __name__ == "__main__":
    B, D, Dh =  1, 4096, 128
    Sq, Skv = 1, 600
    num_heads = 32
    num_kv_heads = 32
    
    head_dim = D // num_heads

    o_shape = (B, Sq, D)

    net = FlashDecodeAttentionNet(
        func="FlashDecodeAttention",
        o_shape=o_shape,
    )

    query_states = Tensor(ms.numpy.randn(B, num_heads, Sq, Dh), mstype.bfloat16)
    key_states = Tensor(ms.numpy.randn(B, num_heads, Skv, Dh), mstype.bfloat16)
    value_states = Tensor(ms.numpy.randn(B, num_heads, Skv, Dh), mstype.bfloat16)
    o_weights = Tensor(ms.numpy.randn(D, D), mstype.bfloat16)
    
    profiler.start()
    ms.runtime.synchronize()        
    t0 = time.perf_counter()
    for _ in range(1):
        attn_output = net(query_states, key_states, value_states, o_weights)
        
    ms.runtime.synchronize()        
    t1 = time.perf_counter()
    profiler.stop()
    profiler.analyse()
    
    memory_allocated = ms.runtime.max_memory_allocated()/10**9
    memory_reserved = ms.runtime.max_memory_reserved()/10**9
    print(f"Memory allocated: {memory_allocated} GB, Memory reserved: {memory_reserved} GB")
    
    print("Ops shape:", attn_output.shape, "Time:", t1-t0)
    print(attn_output)
    
    # # Eager calcs:
    # ms.runtime.synchronize()        
    # t0 = time.perf_counter()
    # attn_output_eager = eager_calcs(query_states, key_states, value_states, o_weights)
    
    # ms.runtime.synchronize()        
    # t1 = time.perf_counter()
    # print("Eager shape:", attn_output_eager.shape, "Time:", t1-t0)
    # print(attn_output_eager)
