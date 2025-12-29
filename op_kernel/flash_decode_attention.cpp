#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
// #include "lib/matmul_intf.h"
using namespace AscendC;

#define BUFFER_NUM 2
#define NEG_INF_F32 -1e29f

class FlashDecodeAttention {
public:
    __aicore__ inline FlashDecodeAttention() {}

    __aicore__ inline void Init(
        GM_ADDR query_states,
        GM_ADDR key_states,
        GM_ADDR value_states,
        GM_ADDR o_weights,
        GM_ADDR attn_output,
        GM_ADDR workspace,
        const FlashDecodeAttentionTilingData& tiling)
    {
        B_   = tiling.B;
        Hq_  = tiling.Hq;
        Hkv_ = tiling.Hkv;
        Sq_  = tiling.Sq;
        Skv_ = tiling.Skv;
        Dh_  = tiling.Dh;
        Dmodel_ = tiling.Dmodel;
        block_dim_ = tiling.block_dim;

        // in MVP, each block only hanlde one head
        heads_per_block_ = tiling.heads_per_block;
        
        block_size_ = tiling.block_size;
        inv_sqrt_dh_ = tiling.inv_sqrt_dh;

        block_idx_ = GetBlockIdx();
        total_heads_ = B_ * Hq_;
        total_blocks_ = (total_heads_ + heads_per_block_ - 1) / heads_per_block_;

       

        // query: [B, Hq, Sq, Dh]
        // key/value: [B, Hkv, Skv, Dh]
        // out: [B, Hq, Sq, Dh]
        qGm_.SetGlobalBuffer((__gm__ bfloat16_t*)query_states);
        kGm_.SetGlobalBuffer((__gm__ bfloat16_t*)key_states);
        vGm_.SetGlobalBuffer((__gm__ bfloat16_t*)value_states);
        wOGm_.SetGlobalBuffer((__gm__ bfloat16_t*)o_weights);
        outGm_.SetGlobalBuffer((__gm__ bfloat16_t*)attn_output);

        // workspaceGm_.SetGlobalBuffer((__gm__ uint8_t*)workspace, 0);

        pipe_.InitBuffer(qBuf_, Dh_ * sizeof(float)); // q_fp32 (Dh float)
        pipe_.InitBuffer(kBuf_, block_size_ * Dh_ * sizeof(float)); // K block
        pipe_.InitBuffer(vBuf_, block_size_ * Dh_ * sizeof(float)); // V block

        pipe_.InitBuffer(qInBuf_, Dh_ * sizeof(bfloat16_t)); // 
        pipe_.InitBuffer(kInBuf_, block_size_ * Dh_ * sizeof(bfloat16_t)); // K block
        pipe_.InitBuffer(vInBuf_, block_size_ * Dh_ * sizeof(bfloat16_t)); // V block
        pipe_.InitBuffer(outBuf_, Dh_ * sizeof(bfloat16_t));

        pipe_.InitBuffer(scoreBuf_, block_size_ * sizeof(float)); // scores fp32
        pipe_.InitBuffer(oBuf_, Dh_ * sizeof(float));          // o fp32
        pipe_.InitBuffer(tmpBuf_, Dh_ * sizeof(float));        // tmp fp32 (optional)
        pipe_.InitBuffer(kfBuf, Dh_ * sizeof(float));

        pipe_.InitBuffer(prodBuf, Dh_ * sizeof(float));
        pipe_.InitBuffer(redWorkBuf, Dh_ * sizeof(float));
        pipe_.InitBuffer(redResBuf, 1 * sizeof(float));

        // pipe.InitBuffer(inQueueW, BUFFER_NUM, D * sizeof(bfloat16_t));    // w_row[D]
        // pipe.InitBuffer(outQueueR, BUFFER_NUM, 1 * sizeof(float));   
    }

    __aicore__ inline void Process(){
        // for safety
        block_idx_ = GetBlockIdx();
        head_begin_ = block_idx_ * heads_per_block_;
        head_end_ = head_begin_ + heads_per_block_;
        if (head_end_ > total_heads_) head_end_ = total_heads_;

        if (block_idx_ >= total_blocks_) return;

        for (uint32_t gh = head_begin_; gh < head_end_; ++gh) {
            uint32_t b = gh / Hq_;
            uint32_t hq = gh - b * Hq_;

            uint32_t hk = MapKvHead(hq);

            for (uint32_t s = 0; s < Sq_; ++s) {
                ComputeOne(b, hq, hk, s);
            }
        }
    }

private:
    __aicore__ inline uint32_t MapKvHead(uint32_t hq) const{
        if (Hkv_ == Hq_) return hq;
        // mod == 0 is already insured
        uint32_t groups = Hq_ / Hkv_;
        return hq / groups;
    }

    __aicore__ inline void ComputeOne(uint32_t b, uint32_t hq, uint32_t hk, uint32_t s){
        // --- 1) Load q[b,hq,s,:] to UB and cast to fp32 ---
        LocalTensor<float> q_fp32 = qBuf_.AllocTensor<float>(); 
        LoadQToFp32(q_fp32, b, hq, s);

        // --- 2) Init running softmax state (m, l, o[Dh]) ---
        float m = NEG_INF_F32;
        float l = 0.0f;

        LocalTensor<float> o_fp32 = oBuf_.AllocTensor<float>();  // size Dh
        Duplicate(o_fp32, 0.0f, Dh_);
        PRINTF("block : %d, Dh_: %d\n", block_idx_, Dh_);
        // for (uint32_t o = 0; o < Dh_; ++o) o_fp32.SetValue(o, 0.0f);
        
        LocalTensor<float> k_blk = kBuf_.AllocTensor<float>(); // [block_size, Dh]
        LocalTensor<float> v_blk = vBuf_.AllocTensor<float>();
        LocalTensor<float> scores = scoreBuf_.AllocTensor<float>(); // [block_size]
        LocalTensor<bfloat16_t> o_bf16 = outBuf_.AllocTensor<bfloat16_t>();

        // --- 3) Scan KV in blocks of block_size_ ---
        for (uint32_t t0 = 0; t0 < Skv_; t0 += block_size_) {
            uint32_t tN = block_size_;
            if (t0 + tN > Skv_) tN = Skv_ - t0;

            // 3.1 load K/V blocks to UB (bfloat16_t/bf16)
            LoadKVBlock(k_blk, v_blk, b, hk, t0, tN);

            // 3.2 compute scores[tN] in fp32 and find block max
            float blk_max = ComputeScoresAndMax(scores, q_fp32, k_blk, tN);

            // 3.3 update running max and rescale old accumulators
            float m_new = (m > blk_max) ? m : blk_max;
            float scale = ExpScalar(m - m_new); // exp(m - m_new) <= 1

            l *= scale;
            Muls(o_fp32, o_fp32, scale, Dh_);

            // 3.4 accumulate this block
            // p = exp(scores[j] - m_new)
            AccumulateBlock(o_fp32, l, scores, v_blk, tN, m_new);

            m = m_new;
        }

        // --- 4) Normalize and store out[b,hq,s,:] ---
        float inv_l = 1.0f / l;
        Muls(o_fp32, o_fp32, inv_l, (int32_t)Dh_);

        // // fp32 -> bf16
        Cast(o_bf16, o_fp32, RoundMode::CAST_ROUND, (int32_t)Dh_);
    
        DataCopy(outGm_[(uint64_t)((b * Hq_ + hq) * Sq_ + s) * Dh_], o_bf16, (int32_t)Dh_);

        qBuf_.FreeTensor(q_fp32);
        oBuf_.FreeTensor(o_fp32);
        kBuf_.FreeTensor(k_blk);
        vBuf_.FreeTensor(v_blk);
        scoreBuf_.FreeTensor(scores);
        outBuf_.FreeTensor(o_bf16);
    }

private:
    __aicore__ inline void LoadQToFp32(LocalTensor<float>& q_fp32,
                                      uint32_t b, uint32_t hq, uint32_t s){
        // GM offset for q: (((b*Hq + hq)*Sq + s)*Dh)
        uint32_t offset = ((b * Hq_ + hq) * Sq_ + s) * Dh_;
        LocalTensor<bfloat16_t> q_bf16 = qInBuf_.AllocTensor<bfloat16_t>();
        DataCopy(q_bf16, qGm_[offset], Dh_);
        pipe_barrier(PIPE_ALL);
        Cast(q_fp32, q_bf16, RoundMode::CAST_NONE, (int32_t)Dh_); // Upcast
        qInBuf_.FreeTensor(q_bf16);
    }

    __aicore__ inline void LoadKVBlock(
        LocalTensor<float>& k_blk,
        LocalTensor<float>& v_blk,
        uint32_t b, uint32_t hk,
        uint32_t t0, uint32_t tN){
        // GM offset for K/V: (((b*Hkv + hk)*Skv + (t0+j))*Dh)
        // Each row is Dh contiguous.
        // DataCopy 2D (tN x Dh) into UB is ideal; else loop rows.
        // placeholder
        uint32_t offset = ((b * Hkv_ + hk) * Skv_ + t0) * Dh_;
        uint32_t length = tN * Dh_;
        LocalTensor<bfloat16_t> k_bf16 = kInBuf_.AllocTensor<bfloat16_t>();
        LocalTensor<bfloat16_t> v_bf16 = vInBuf_.AllocTensor<bfloat16_t>();
        
        DataCopy(k_bf16, kGm_[offset], length);
        DataCopy(v_bf16, vGm_[offset], length);
        pipe_barrier(PIPE_ALL);
        Cast(k_blk, k_bf16, RoundMode::CAST_NONE, (int32_t)length); // Upcast
        Cast(v_blk, v_bf16, RoundMode::CAST_NONE, (int32_t)length); // Upcast
        kInBuf_.FreeTensor(k_bf16);
        vInBuf_.FreeTensor(v_bf16);
    }

    __aicore__ inline float ComputeScoresAndMax(LocalTensor<float>& scores,
        const LocalTensor<float>& q_fp32,
        const LocalTensor<float>& k_blk,
        uint32_t tN){
        // scores: [tN]
        // q_fp32: [Dh]
        // k_blk: [tN, Dh]
        LocalTensor<float> prod = prodBuf.AllocTensor<float>();    // [D]
        LocalTensor<float> work = redWorkBuf.AllocTensor<float>();    // [D]
        LocalTensor<float> redRes = redResBuf.AllocTensor<float>(); // [1]
        LocalTensor<float> k_piece = kfBuf.AllocTensor<float>();    // [Dh]

        for(int j = 0; j < tN; j++){
            // Copy(k_piece, , (int32_t)Dh_);
            Mul(prod, q_fp32, k_blk[j * Dh_], (int32_t)Dh_);
            ReduceSum(redRes, prod, work, (int32_t)Dh_);
            scores.SetValue(j, redRes.GetValue(0) * inv_sqrt_dh_);
        }

        ReduceMax(redRes, scores, work, (int32_t)tN);
        float blk_max = redRes.GetValue(0);

        redResBuf.FreeTensor(redRes);
        prodBuf.FreeTensor(prod);
        redWorkBuf.FreeTensor(work);

        return blk_max;
    }

    __aicore__ inline float ExpScalar(float x){
        LocalTensor<float> redRes = redResBuf.AllocTensor<float>(); 
        redRes.SetValue(0, x);
        Exp(redRes, redRes, (int32_t)1);
        float ans = redRes.GetValue(0);
        redResBuf.FreeTensor(redRes);
        return ans; 
    }

    __aicore__ inline void AccumulateBlock(LocalTensor<float>& o_fp32,
        float& l,
        LocalTensor<float>& scores,
        LocalTensor<float>& v_blk,
        uint32_t tN,
        float m_new){
        
        LocalTensor<float> redRes = redResBuf.AllocTensor<float>(); // [1]
        LocalTensor<float> work = redWorkBuf.AllocTensor<float>();    // [D]
        
        Adds(scores, scores, -m_new, (int32_t)tN);
        Exp(scores, scores, (int32_t)tN);
        
        ReduceSum(redRes, scores, work, (int32_t)tN);
        l += redRes.GetValue(0);
        
        for (uint32_t j = 0; j < tN; ++j) {
            float pj = scores.GetValue(j);
            Axpy(o_fp32, v_blk[j * Dh_], pj, (int32_t)Dh_);
        }
        redResBuf.FreeTensor(redRes);
        redWorkBuf.FreeTensor(work);
    }

private:
    uint32_t B_{0}, Hq_{0}, Hkv_{0}, Sq_{0}, Skv_{0}, Dh_{0}, Dmodel_{0};
    uint32_t block_dim_{0}, heads_per_block_{0}, block_size_{0};
    float inv_sqrt_dh_{0.0f};

    uint32_t block_idx_{0};
    uint32_t total_heads_{0}, total_blocks_{0};
    uint32_t head_begin_{0}, head_end_{0};

    // ---- global tensors ----
    GlobalTensor<bfloat16_t> qGm_;
    GlobalTensor<bfloat16_t> kGm_;
    GlobalTensor<bfloat16_t> vGm_;
    GlobalTensor<bfloat16_t> wOGm_;
    GlobalTensor<bfloat16_t> outGm_;
    GlobalTensor<uint8_t> workspaceGm_;

    // ---- UB buffers / pipe ----
    TPipe pipe_;

    // UB buffers (adjust sizes based on your Dh/block_size)
    TBuf<TPosition::VECCALC> qBuf_;      // Dh * fp32
    TBuf<TPosition::VECCALC> kBuf_;      // block_size * Dh * bfloat16_t
    TBuf<TPosition::VECCALC> vBuf_;      // block_size * Dh * bfloat16_t
    TBuf<TPosition::VECCALC> scoreBuf_;  // block_size * fp32
    TBuf<TPosition::VECCALC> oBuf_;      // Dh * fp32
    TBuf<TPosition::VECCALC> tmpBuf_;    // Dh * fp32 (optional)

    TBuf<TPosition::VECCALC> kfBuf, prodBuf, redWorkBuf, redResBuf;

    TBuf<TPosition::VECIN>  qInBuf_, kInBuf_, vInBuf_;
    TBuf<TPosition::VECOUT> outBuf_;   // 如果你的版本区分输出
    TBuf<TPosition::VECCALC> prodBuf_, redWorkBuf_, redResBuf_;


    // TQue<QuePosition::VECIN,  BUFFER_NUM> inQueueW;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueR;
};

    
extern "C" __global__ __aicore__
void flash_decode_attention(
    GM_ADDR query_states,
    GM_ADDR key_states,
    GM_ADDR value_states,
    GM_ADDR o_weights,
    GM_ADDR attn_output,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    FlashDecodeAttention op;
    op.Init(query_states, key_states, value_states, o_weights, attn_output, workspace, tiling_data);
    op.Process();
}