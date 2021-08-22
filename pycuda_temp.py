import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpu
import pycuda.autoinit
import SimpleITK as sitk
from PIL import Image
import math
import time
import os
import numpy as np

_path = r'E:\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64'
if os.system("cl.exe"):
    os.environ['PATH'] += ';' + _path
if os.system("cl.exe"):
    raise RuntimeError("cl.exe still not found, path probably incorrect")



mod = SourceModule("""
__global__ void fun(float *out, float *a)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if(i*512 + j > 512)
    {
        out[i*512 + j] = (a[i*512 + j] + a[i*512 + j -1] + a[(i-1)*512 + j]) / 3;
    }
}
""")
img = sitk.ReadImage('lobe512_000_0000.nii.png')
data = sitk.GetArrayFromImage(img)
a = data.astype(np.float32)

out = np.empty_like(a)
func = mod.get_function("fun")

blockspergrid_x = int(math.ceil(512 / 16))
blockspergrid_y = int(math.ceil(512 / 16))
gpu_start = time.time()
func(cuda.Out(out), cuda.In(a), block=(16, 16, 1), grid=(blockspergrid_x, blockspergrid_y))
gpu_end = time.time()
gpu_time = gpu_end - gpu_start
print('GPU time:' + str(gpu_time))
im = Image.fromarray(out)
im.convert('L').save("temp_gpu.png")
