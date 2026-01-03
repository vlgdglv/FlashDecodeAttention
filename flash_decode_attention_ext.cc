#include "ms_extension.h"

namespace mindspore::pyboost {

// 1) 定义一个前向 Function
class FlashDecodeAttentionFunc : public Function {
 public:
  // inputs: q, k, v, w
  BaseTensorPtrList Forward(const BaseTensorPtrList &inputs, AutogradContext *ctx) override {
    // 2) 推导输出：先写死 (1,1,4096) 跑通
    auto out = std::make_shared<BaseTensor>();
    out->set_shape({1, 1, 4096});
    out->set_dtype(inputs[0]->dtype());

    // 3) 调用 aclnn：用官方提供的 CustomLaunchAclnn（2.6 文档明确存在）
    // 伪代码：具体参数组织按你 op 的 aclnn 接口要求来
    CustomLaunchAclnn("FlashDecodeAttention", /*inputs=*/inputs, /*outputs=*/{out});

    return {out};
  }

};

// 4) pybind 导出一个 Python 可调用入口
py::object flash_decode_attention_py(const py::object &q,
                                    const py::object &k,
                                    const py::object &v,
                                    const py::object &w) {

}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("flash_decode_attention", &flash_decode_attention_py, "FlashDecodeAttention");
}

}  // namespace mindspore::pyboost
