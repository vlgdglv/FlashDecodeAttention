import os
import mindspore as ms
from mindspore.ops import CustomOpBuilder

def load_flash_decode_ext():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(this_dir, "flash_decode_attention_ext.cc")

    builder = CustomOpBuilder(
        name="flash_decode_attention_ext",
        sources=src,
        backend="Ascend",
    )
    mod = builder.load()
    return mod

if __name__ == "__main__":
    mod = load_flash_decode_ext()
