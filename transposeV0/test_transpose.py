import torch
from torch.utils.cpp_extension import load

transpose_ext = load(
    name="transpose_cuda",
    sources=["transpose_cuda_kernel.cu"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)

# 构建输入数据
input_tensor = torch.randn(128, 64, device='cuda')
output_tensor = torch.empty(64, 128, device='cuda')

# 调用 CUDA kernel
transpose_ext.transpose_shared(input_tensor, output_tensor)

# 验证正确性
expected = input_tensor.t()
diff = (output_tensor - expected).abs().max()
print("Max difference:", diff.item())
