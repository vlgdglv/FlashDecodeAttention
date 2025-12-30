import os, time
import numpy as np
import mindspore as ms
from mindnlp.core import ops as core_ops 

import math
from mindnlp.core import nn
from mindspore.nn import Cell
from mindspore.ops import CustomRegOp, DataType

from mindspore import context, Tensor
import mindspore.ops as ops
from mindspore.ops import DataType, CustomRegOp
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype


NUM_HEADS = 32
HEAD_DIM = 128
INPUT_TYPE = "sep"

def check_test_data():
    test_data_dir = "/home/ma-user/work/task1/task1/test_data/sample_flashAttn"
    
    test_activation_dir = "/home/ma-user/work/task1/test_data/sample_flashAttn/activation"
    layer_idx, sample_idx = 0, 0
    
    print("========= check test data activation ========= ")
    
    attn_output = np.load(os.path.join(test_activation_dir, f"Janus_Pro_7B_layer{layer_idx}_sample{sample_idx}_attn_output.npy"))
    query_states = np.load(os.path.join(test_activation_dir,  f"Janus_Pro_7B_layer{layer_idx}_sample{sample_idx}_query_states.npy"))
    key_states = np.load(os.path.join(test_activation_dir,  f"Janus_Pro_7B_layer{layer_idx}_sample{sample_idx}_key_states.npy"))
    value_states = np.load(os.path.join(test_activation_dir, f"Janus_Pro_7B_layer{layer_idx}_sample{sample_idx}_value_states.npy"))
    
    # print(hidden_states.dtype, query_states.dtype, key_states.dtype, value_states.dtype)
    # print(hidden_states.shape, query_states.shape, key_states.shape, value_states.shape)
    
    attn_output_ms = ms.Tensor(attn_output, ms.bfloat16) # [1, SEQ_LEN, HIDDEN_DIM]
    query_states_ms = ms.Tensor(query_states, ms.bfloat16)   # [1, HEAD_NUM, SEQ_LEN, HEAD_DIM]
    key_states_ms = ms.Tensor(key_states, ms.bfloat16)       # [1, HEAD_NUM, SEQ_LEN, HEAD_DIM]
    value_states_ms = ms.Tensor(value_states, ms.bfloat16)   # [1, HEAD_NUM, SEQ_LEN, HEAD_DIM]
    
    print(attn_output_ms.dtype, query_states_ms.dtype, key_states_ms.dtype, value_states_ms.dtype)
    print(attn_output_ms.shape, query_states_ms.shape, key_states_ms.shape, value_states_ms.shape)


    test_weights_dir = "/home/ma-user/work/task1/test_data/sample_flashAttn/weight"
    print("========= check test data activation ========= ")

    o_proj_weight = np.load(os.path.join(test_weights_dir, f"Janus_Pro_7B_layer{layer_idx}_o_proj_weight.npy"))
    
    o_proj_weight = ms.Tensor(o_proj_weight, ms.bfloat16)    # [HIDDEN_DIM, HIDDEN_DIM]
    
    print(o_proj_weight.dtype)
    print(o_proj_weight.shape)


class FlashDeodeTester:
    def __init__(
        self,
        base_dir: str,
        num_layers: int,
        sample_idx: int = 0,
        dtype=ms.bfloat16,
    ):
        self.base_dir = base_dir
        self.activation_dir = os.path.join(base_dir, "activation")
        self.weight_dir = os.path.join(base_dir, "weight")
        self.num_layers = num_layers
        self.sample_idx = sample_idx
        self.dtype = dtype

        # data[layer_idx] = {
        #   "hidden_states", "query_states", "key_states", "value_states",
        #   "q_proj_weight", "k_proj_weight", "v_proj_weight"
        # }
        self.data = {}
        self._load_all_layers()

    def _load_one_layer(self, layer_idx: int):
        sample_idx = self.sample_idx
        prefix = f"Janus_Pro_7B_layer{layer_idx}_sample{sample_idx}"

        # activation
        attn_output = np.load(
            os.path.join(self.activation_dir, f"{prefix}_attn_output.npy")
        )
        query_states = np.load(
            os.path.join(self.activation_dir, f"{prefix}_query_states.npy")
        )
        key_states = np.load(
            os.path.join(self.activation_dir, f"{prefix}_key_states.npy")
        )
        value_states = np.load(
            os.path.join(self.activation_dir, f"{prefix}_value_states.npy")
        )
        o_proj_weight = np.load(
            os.path.join(self.weight_dir, f"Janus_Pro_7B_layer{layer_idx}_o_proj_weight.npy")
        )

        attn_output_ms = ms.Tensor(attn_output, self.dtype)
        query_states_ms = ms.Tensor(query_states, self.dtype)
        key_states_ms = ms.Tensor(key_states, self.dtype)
        value_states_ms = ms.Tensor(value_states, self.dtype)

        o_proj_weight = ms.Tensor(o_proj_weight, self.dtype)

        self.data[layer_idx] = {
            "attn_output": attn_output_ms,
            "query_states": query_states_ms,
            "key_states": key_states_ms,
            "value_states": value_states_ms,
            "o_proj_weight": o_proj_weight,
        }

    def _load_all_layers(self):
        print("========= loading test data =========")
        for layer_idx in range(self.num_layers):
            self._load_one_layer(layer_idx)
            d = self.data[layer_idx]
            print(
                f"Layer {layer_idx}: o_proj_weight {d['o_proj_weight'].shape}, "
                f"q {d['query_states'].shape}, k {d['key_states'].shape}, v {d['value_states'].shape}",
                end="\r"
            )

    def run_func(self, func, WARMUP=3, REPEAT=10):
        """
            func input: (hidden_states, q_proj_weight, k_proj_weight, v_proj_weight)
            func output: (q_pred, k_pred, v_pred)
        """
        preds = {}
        exec_time = []
        for layer_idx, d in self.data.items():
            query_states = d["query_states"]
            key_states = d["key_states"]
            value_states = d["value_states"]
            o_proj_weight = d["o_proj_weight"]
            
            for i in range(WARMUP):
                _ = func(query_states, key_states, value_states, o_proj_weight)
                if i < 10:
                    ms.runtime.synchronize()
            ms.runtime.synchronize()
                
            for _ in range(REPEAT):
                t0 = time.perf_counter()
                attn_output = func(query_states, key_states, value_states, o_proj_weight)
                ms.runtime.synchronize()
                t1 = time.perf_counter()
                exec_time.append(t1 - t0)

            # per_call = (t1 - t0) / REPEAT
            # exec_time.append(per_call)

            preds[layer_idx] = attn_output
        return preds, exec_time

    def evaluate(self, func, atol=1e-3, rtol=1e-3, verbose=False):
    
        preds, exec_time = self.run_func(func)
    
        memory_allocated = ms.runtime.max_memory_allocated()/10**9
        memory_reserved = ms.runtime.max_memory_reserved()/10**9
        
        overall_ok = True

        total_test, pass_test = 0, 0
        mean_diff = .0
        
        for layer_idx, (attn_output) in preds.items():
            if attn_output is None:
                print("Accuracy test for layer {} skipped \r".format(layer_idx))
                continue
            
            d = self.data[layer_idx]
            metrics = {}
            for name, pred, gt in [
                ("attn_output", attn_output, d["attn_output"]),
            ]:
                pred32 = pred.astype(ms.float32).asnumpy()
                gt32 = gt.astype(ms.float32).asnumpy()
                diff = pred32 - gt32
                abs_err = np.abs(diff)
                max_abs = abs_err.max()
                denom = np.abs(gt32).max() + 1e-8
                max_rel = max_abs / denom
                mean_diff += np.abs(diff).mean()
                
                ok = max_abs <= atol + rtol * np.abs(gt32).max()
                metrics[name] = (max_abs, max_rel, ok)
                overall_ok = overall_ok and ok
                if ok:
                    pass_test += 1
                total_test += 1

            if verbose:
                print(f"---- Layer {layer_idx} ----")
                for name in ["attn_output"]:
                    max_abs, max_rel, ok = metrics[name]
                    print(
                        f"{name}: max_abs={max_abs:.3e}, "
                        f"max_rel={max_rel:.3e}, pass={ok}"
                    )
        if overall_ok is not None:
            print(f"Overall pass: {overall_ok}, pass rate: {pass_test}/{total_test}, mean abs diff: {mean_diff.mean()}")
        print(f"Execution time: {1000*np.mean(exec_time):.4f} +- {1000*np.std(exec_time):.4f} ms")
        print("Latency: ", stat_latency(exec_time))
        print(f"Memory allocated: {memory_allocated} GB, Memory reserved: {memory_reserved} GB")
        return overall_ok
    
def stat_latency(ts):
    ts = np.array(ts) * 1000
    return {
        "p50": float(np.percentile(ts, 50)),
        "p90": float(np.percentile(ts, 90)),
        "p99": float(np.percentile(ts, 99)),
        "mean": float(ts.mean()),
        "std": float(ts.std()),
        "max": float(ts.max()),
        "min": float(ts.min()),
    }
    
def eager_attn(query_states, key_states, value_states, o_weights):
    query_states = query_states.to(ms.float32)
    key_states = key_states.to(ms.float32)
    value_states = value_states.to(ms.float32)
    o_weights = o_weights.to(ms.float32)

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
    attn_output = ms.mint.nn.functional.linear(attn_output, o_weights, None)
    return attn_output


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
            query_states, 
            key_states, 
            value_states, 
            o_weights
            )
        # print(">>> after fused_qkv")
        return attn_output
    
opsNet = FlashDecodeAttentionNet(
    func="FlashDecodeAttention",
    o_shape=(1, 1, 4096),
)

def ops_impl(query_states, key_states, value_states, o_weights):
    attn_output = opsNet(query_states, key_states, value_states, o_weights)
    attn_output = ms.mint.nn.functional.linear(attn_output, o_weights, None)
    return attn_output

ms.set_context(
    mode=ms.GRAPH_MODE,
    device_target="Ascend"
    )
print(ms.get_context("mode"))

if __name__ == "__main__":
    for idx in [3, 12, 43]:
        evaluator = FlashDeodeTester(
            base_dir="/home/ma-user/work/task1/test_data/sample_flashAttn", 
            num_layers=30, 
            sample_idx=idx, 
            dtype=ms.bfloat16
        )
        
        # evaluator.evaluate(fused_mint_qkvProjTrans)
        
        print(">>> Custom Ops " * 6)
        evaluator.evaluate(ops_impl)
        
        print(">>> Original Eager" * 6)
        evaluator.evaluate(eager_attn)