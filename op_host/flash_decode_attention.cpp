
#include "flash_decode_attention_tiling.h"
#include "register/op_def_registry.h"
#include <cmath>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context){
    FlashDecodeAttentionTilingData tiling;

    const gert::StorageShape* q_shape_ptr = context->GetInputShape(0);
    const gert::StorageShape* k_shape_ptr = context->GetInputShape(1);
    const gert::StorageShape* v_shape_ptr = context->GetInputShape(2);
    const gert::StorageShape* w_shape_ptr = context->GetInputShape(3);
    
    const gert::Shape& q = q_shape_ptr->GetStorageShape();
    const gert::Shape& k = k_shape_ptr->GetStorageShape();
    const gert::Shape& v = v_shape_ptr->GetStorageShape();
    const gert::Shape& w = w_shape_ptr->GetStorageShape();

    if (q.GetDimNum()!=4 || k.GetDimNum()!=4 || v.GetDimNum()!=4 ){
        return ge::GRAPH_FAILED;
    }

    if (w.GetDimNum()!=2){
        return ge::GRAPH_FAILED;
    }

    uint32_t Bq = static_cast<uint32_t>(q.GetDim(0));
    uint32_t Hq = static_cast<uint32_t>(q.GetDim(1));
    uint32_t Sq = static_cast<uint32_t>(q.GetDim(2));
    uint32_t Dhq = static_cast<uint32_t>(q.GetDim(3));

    uint32_t Bk = static_cast<uint32_t>(k.GetDim(0));
    uint32_t Hk = static_cast<uint32_t>(k.GetDim(1));
    uint32_t Sk = static_cast<uint32_t>(k.GetDim(2));
    uint32_t Dhk = static_cast<uint32_t>(k.GetDim(3));

    uint32_t Bv = static_cast<uint32_t>(v.GetDim(0));
    uint32_t Hv = static_cast<uint32_t>(v.GetDim(1));
    uint32_t Sv = static_cast<uint32_t>(v.GetDim(2));
    uint32_t Dhv = static_cast<uint32_t>(v.GetDim(3));

    uint32_t Wo0 = static_cast<uint32_t>(w.GetDim(0));
    uint32_t Wo1 = static_cast<uint32_t>(w.GetDim(1));

    if (Bq != Bk || Bq != Bv) return ge::GRAPH_FAILED;
    if (Dhq != Dhk || Dhq != Dhv) return ge::GRAPH_FAILED;
    if (Hk != Hv) return ge::GRAPH_FAILED;
    if (Sk != Sv) return ge::GRAPH_FAILED;
    
    // GQA check
    if (Hq==0 || Hk==0 || Hv==0) return ge::GRAPH_FAILED;
    if (Hq % Hk != 0) return ge::GRAPH_FAILED;

    uint32_t Dmodel = Wo0;

    tiling.set_B(Bq);
    tiling.set_Hq(Hq);
    tiling.set_Sq(Sq);
    tiling.set_Dh(Dhq);
    tiling.set_Hkv(Hk);
    tiling.set_Skv(Sk);
    tiling.set_Dmodel(Dmodel);

    uint32_t block_size = 64;
    tiling.set_block_size(block_size);

    uint32_t total_heads = Bq * Hq;
    uint32_t block_dim = std::min<uint32_t>(32, total_heads);

    tiling.set_block_dim(block_dim);
    tiling.set_heads_per_block(1);

    context->SetBlockDim(block_dim);

    // the famous 1/sqrt(Dh)
    float inv_sqrt_dh = 1.0f / std::sqrt(Dhq);
    tiling.set_inv_sqrt_dh(inv_sqrt_dh);
    
    std::cout << "[TilingFunc]:" << std::endl;
    std::cout << "B: " << Bq << std::endl;
    std::cout << "Hq: " << Hq << std::endl;
    std::cout << "Sq: " << Sq << std::endl;
    std::cout << "Dhq: " << Dhq << std::endl;
    std::cout << "Hk: " << Hk << std::endl;
    std::cout << "Sk: " << Sk << std::endl;
    std::cout << "Dhk: " << Dhk << std::endl;
    std::cout << "Hv: " << Hv << std::endl;
    std::cout << "Sv: " << Sv << std::endl;
    std::cout << "Dhv: " << Dhv << std::endl;
    std::cout << "Wo0: " << Wo0 << std::endl;
    std::cout << "Wo1: " << Wo1 << std::endl;
    std::cout << "Dmodel: " << Dmodel << std::endl;
    std::cout << "inv_sqrt_dh: " << inv_sqrt_dh << std::endl;

    size_t* workspaces = context->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // workspaces[0] =  systemWorkspaceSize + userWorkspaceSize;
    workspaces[0] = 0;

    auto *raw = context->GetRawTilingData();
    tiling.SaveToBuffer(raw->GetData(), raw->GetCapacity());
    raw->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context){
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context){
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class FlashDecodeAttention : public OpDef {
public:
    explicit FlashDecodeAttention(const char* name) : OpDef(name)
    {
        this->Input("query_states")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("key_states")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("value_states")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("o_weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("attn_output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(FlashDecodeAttention);
}
