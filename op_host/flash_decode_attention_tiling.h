// tilingdata.h
#include "register/tilingdata_base.h"
#include <cstdint>

namespace optiling {

BEGIN_TILING_DATA_DEF(FlashDecodeAttentionTilingData)
  // q: [B, Hq, Sq, Dh]
  TILING_DATA_FIELD_DEF(uint32_t, B);
  TILING_DATA_FIELD_DEF(uint32_t, Hq);
  TILING_DATA_FIELD_DEF(uint32_t, Sq);
  TILING_DATA_FIELD_DEF(uint32_t, Dh);

  // kv: [B, Hkv, Skv, Dh]
  TILING_DATA_FIELD_DEF(uint32_t, Hkv);
  TILING_DATA_FIELD_DEF(uint32_t, Skv);

  // o_weights: [Dmodel, Dmodel]
  TILING_DATA_FIELD_DEF(uint32_t, Dmodel);

  // scheduling / tiling
  TILING_DATA_FIELD_DEF(uint32_t, block_dim);
  TILING_DATA_FIELD_DEF(uint32_t, heads_per_block);
  TILING_DATA_FIELD_DEF(uint32_t, block_size);

  // precompute constants (optional)
  TILING_DATA_FIELD_DEF(float, inv_sqrt_dh);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FlashDecodeAttention, FlashDecodeAttentionTilingData)

} // namespace optiling
