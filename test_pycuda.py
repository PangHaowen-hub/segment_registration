import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import os
import numpy as np
import pycuda.compiler as nvcc
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu
import pycuda.autoinit

_path = r'E:\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64'
if os.system("cl.exe"):
    os.environ['PATH'] += ';' + _path
if os.system("cl.exe"):
    raise RuntimeError("cl.exe still not found, path probably incorrect")

mod = SourceModule("""
__global__ void doublify(float *a)
{
int idx = threadIdx.x + threadIdx.y*4;
a[idx] *= 2;
}
""")

a = np.random.randn(16, 16)
a = a.astype(np.float32)
a_gpu = cuda.mem_alloc(a.nbytes)  # 在GPU上为a分配所需的显存
cuda.memcpy_htod(a_gpu, a)  # 将数据转移到 GPU

func = mod.get_function("doublify")
func(a_gpu, block=(16, 16, 1))

a_doubled = np.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)
print(a_doubled)
print(a)
