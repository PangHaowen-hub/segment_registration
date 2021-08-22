import math
import numpy as np
from tqdm import trange
from tqdm import tqdm
import SimpleITK as sitk
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import os

_path = r'E:\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64'
if os.system("cl.exe"):
    os.environ['PATH'] += ';' + _path
if os.system("cl.exe"):
    raise RuntimeError("cl.exe still not found, path probably incorrect")


class Cluster(object):
    cluster_index = 1

    def __init__(self, h, w, c, l=0):
        self.update(h, w, c, l)
        self.pixels = []
        self.no = self.cluster_index
        Cluster.cluster_index += 1

    # l表示亮度,0（黑色）到100（白色）
    # a表示从洋红色至绿色的范围（a为负值指示绿色而正值指示品红）
    # b表示从黄色至蓝色的范围（b为负值指示蓝色而正值指示黄色）
    def update(self, h, w, c, l):
        self.h = h
        self.w = w
        self.c = c
        self.l = l

    def __str__(self):
        return "{},{},{}:{} ".format(self.h, self.w, self.c, self.l)

    def __repr__(self):
        return self.__str__()


class SLICProcessor(object):

    def make_cluster(self, h, w, c):
        h = int(h)
        w = int(w)
        c = int(c)
        return Cluster(h, w, c, self.data[h][w][c])

    def __init__(self, filename, K, M):
        self.K = K  # 预分割为K个相同尺寸的超像素
        self.M = M
        self.img = sitk.ReadImage(filename)
        self.data = sitk.GetArrayFromImage(self.img)
        self.image_height = self.data.shape[0]
        self.image_width = self.data.shape[1]
        self.image_channel = self.data.shape[2]
        self.N = self.image_height * self.image_width * self.image_channel  # 图片总共有N个像素点
        self.S = int(math.pow(self.N / self.K, 1 / 3))  # 每个超像素的大小为N/K,相邻种子点的距离（步长）近似为S = sqrt(N / K)
        self.clusters = []
        self.label = {}
        self.dis = np.full((self.image_height, self.image_width, self.image_channel), np.inf)

    def init_clusters(self):  # 生成种子点
        h = self.S / 2
        w = self.S / 2
        c = self.S / 2

        while h < self.image_height:
            while w < self.image_width:
                while c < self.image_channel:
                    self.clusters.append(self.make_cluster(h, w, c))
                    c += self.S
                c = self.S / 2
                w += self.S
            w = self.S / 2
            h += self.S

    def get_gradient(self, h, w, c):  # 计算当前位置梯度
        if w + 1 >= self.image_width:
            w = self.image_width - 2
        if h + 1 >= self.image_height:
            h = self.image_height - 2
        if c + 1 >= self.image_channel:
            c = self.image_channel - 2

        gradient = int(self.data[h + 1][w + 1][c + 1]) - int(self.data[h][w][c])
        return gradient

    def move_clusters(self):
        # 在种子点的n*n邻域内重新选择种子点（一般取n=3）。
        # 具体方法为：计算该邻域内所有像素点的梯度值，将种子点移到该邻域内梯度最小的地方。
        # 这样做的目的是为了避免种子点落在梯度较大的轮廓边界上，以免影响后续聚类效果。
        for cluster in self.clusters:
            cluster_gradient = self.get_gradient(cluster.h, cluster.w, cluster.c)
            for dh in range(-1, 2):
                for dw in range(-1, 2):
                    for dc in range(-1, 2):
                        _h = cluster.h + dh
                        _w = cluster.w + dw
                        _c = cluster.c + dc
                        new_gradient = self.get_gradient(_h, _w, _c)
                        if new_gradient < cluster_gradient:
                            cluster.update(_h, _w, _c, self.data[_h][_w][_c])
                            cluster_gradient = new_gradient

    def assignment(self):  # 耗时最长
        mod = SourceModule("""
        __global__ void doublify(int *L_gpu,int *Dc_gpu, int *h_gpu, int *w_gpu, int *c_gpu, 
        int *cluster_h_gpu, int *cluster_w_gpu, int *cluster_c_gpu, int *cluster_dis_gpu)
        {
        int idx = threadIdx.x + threadIdx.y * blcokDim.x;
        a[idx] *= 2;
        }
        """)
        # int Ds = math.sqrt(math.pow(h_gpu - cluster_h_gpu, 2) +
        # math.pow(w_gpu - cluster_w_gpu, 2) +
        # math.pow(c_gpu - cluster_c_gpu, 2))
        # cluster_dis_gpu = math.sqrt(math.pow(Dc_gpu / self.M, 2) + math.pow(Ds / self.S, 2))
        # }
        # """)
        for cluster in tqdm(self.clusters):
            for h in range(cluster.h - 2 * self.S, cluster.h + 2 * self.S):
                if h < 0 or h >= self.image_height:
                    continue
                for w in range(cluster.w - 2 * self.S, cluster.w + 2 * self.S):
                    if w < 0 or w >= self.image_width:
                        continue
                    for c in range(cluster.c - 2 * self.S, cluster.c + 2 * self.S):
                        if c < 0 or c >= self.image_channel:
                            continue
                        L = self.data[h][w][c]
                        L_gpu = cuda.mem_alloc(L)  # 在GPU上为a分配所需的显存
                        cuda.memcpy_htod(L_gpu, L)  # 将数据转移到 GPU
                        Dc = abs(int(L) - int(cluster.l))
                        Dc_gpu = cuda.mem_alloc(Dc)
                        cuda.memcpy_htod(Dc_gpu, Dc)
                        h_gpu = cuda.mem_alloc(h)
                        cuda.memcpy_htod(h_gpu, h)
                        w_gpu = cuda.mem_alloc(w)
                        cuda.memcpy_htod(w_gpu, w)
                        c_gpu = cuda.mem_alloc(c)
                        cuda.memcpy_htod(c_gpu, c)
                        cluster_h_gpu = cuda.mem_alloc(cluster.h)
                        cuda.memcpy_htod(cluster_h_gpu, cluster.h)
                        cluster_w_gpu = cuda.mem_alloc(cluster.w)
                        cuda.memcpy_htod(cluster_w_gpu, cluster.w)
                        cluster_c_gpu = cuda.mem_alloc(cluster.c)
                        cuda.memcpy_htod(cluster_c_gpu, cluster.c)
                        cluster_dis_gpu = cuda.mem_alloc(self.dis[h][w][c])
                        cuda.memcpy_htod(cluster_dis_gpu, self.dis[h][w][c])
                        func = mod.get_function("doublify")
                        func(L_gpu, Dc_gpu, h_gpu, w_gpu, c_gpu, cluster_h_gpu, cluster_w_gpu, cluster_c_gpu,
                             cluster_dis_gpu, block=(16, 16, 1))
                        D = np.empty_like(self.dis[h][w][c])
                        cuda.memcpy_dtoh(D, cluster_dis_gpu)
                        if D < self.dis[h][w][c]:
                            if (h, w, c) not in self.label:
                                self.label[(h, w, c)] = cluster
                                cluster.pixels.append((h, w, c))
                            else:
                                self.label[(h, w, c)].pixels.remove((h, w, c))
                                self.label[(h, w, c)] = cluster
                                cluster.pixels.append((h, w, c))
                            self.dis[h][w][c] = D

    def update_cluster(self):
        for cluster in self.clusters:
            sum_h = sum_w = sum_c = number = 0
            for p in cluster.pixels:
                sum_h += p[0]
                sum_w += p[1]
                sum_c += p[2]
                number += 1
            _h = int(sum_h / number)
            _w = int(sum_w / number)
            _c = int(sum_c / number)
            cluster.update(_h, _w, _c, self.data[_h][_w][_c])

    def save_current_image(self, name):
        image_arr = np.copy(self.data)
        for cluster in self.clusters:
            for p in cluster.pixels:
                image_arr[p[0]][p[1]][p[2]] = cluster.l
            image_arr[cluster.h][cluster.w][cluster.c] = 0
        new_mask_img = sitk.GetImageFromArray(image_arr)
        new_mask_img.SetDirection(self.img.GetDirection())
        new_mask_img.SetOrigin(self.img.GetOrigin())
        new_mask_img.SetSpacing(self.img.GetSpacing())
        sitk.WriteImage(new_mask_img, name)

    def iterate_10times(self):
        self.init_clusters()  # 均匀生成种子点
        self.move_clusters()  # 根据梯度重新选择种子点
        for i in trange(10):
            self.assignment()
            self.update_cluster()
            name = 'lobe_M{m}_K{k}_loop{loop}.nii.gz'.format(loop=i, m=self.M, k=self.K)
            self.save_current_image(name)


if __name__ == '__main__':
    p = SLICProcessor('lobe512_000_0000.nii.gz', 100000, 100)
    p.iterate_10times()
    print('******')
