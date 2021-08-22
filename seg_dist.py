import math
import numpy as np
from tqdm import trange
from tqdm import tqdm
import SimpleITK as sitk


class Cluster(object):
    cluster_index = 1

    def __init__(self, h, w, c, l=0):
        self.update(h, w, c, l)
        self.pixels = []
        self.no = self.cluster_index
        Cluster.cluster_index += 1

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

    def __init__(self, filename, K, M, init_clusters_list):
        self.K = K  # 分为5类
        self.M = M
        self.img = sitk.ReadImage(filename)
        self.data = sitk.GetArrayFromImage(self.img)
        self.image_height = self.data.shape[0]
        self.image_width = self.data.shape[1]
        self.image_channel = self.data.shape[2]
        self.N = self.image_height * self.image_width * self.image_channel  # 图片总共有N个像素点
        self.S = int(math.pow(self.N / self.K, 1 / 3))  # 每个超像素的大小为N/K,相邻种子点的距离（步长）近似为S = sqrt(N / K)
        self.clusters = []
        self.dis = np.full((self.image_height, self.image_width, self.image_channel), np.inf)
        self.init_clusters_list = init_clusters_list

    def init_clusters(self):  # 创建种子点
        for cinit_lusters_pos in self.init_clusters_list:
            self.clusters.append(self.make_cluster(cinit_lusters_pos[0], cinit_lusters_pos[1], cinit_lusters_pos[2]))

    def assignment(self):  # 耗时最长
        cluster_info = []
        for cluster in self.clusters:
            cluster_info.append([cluster.h, cluster.w, cluster.c, cluster.c])
        cluster_info = np.asarray(cluster_info, dtype='int16')
        t0 = np.asarray([[i] * (self.data.shape[1] * self.data.shape[2]) for i in range(self.data.shape[0])],
                        dtype='int16').flatten()
        t1 = np.asarray([[[i] * self.data.shape[1] for i in range(self.data.shape[2])] * self.data.shape[0]],
                        dtype='int16').flatten()
        t2 = np.asarray([list(range(0, self.data.shape[2])) * (self.data.shape[0] * self.data.shape[1])],
                        dtype='int16').flatten()
        t3 = self.data.flatten()
        img_info = np.vstack((t0, t1, t2, t3)).transpose().astype('int32')
        # Ds = np.zeros((cluster_info.shape[0], self.data.shape[0] * self.data.shape[1] * self.data.shape[2], 4), dtype='int32')
        Ds = np.zeros((cluster_info.shape[0], self.data.shape[0] * self.data.shape[1] * self.data.shape[2], 3),
                      dtype='int32')

        for i in trange(cluster_info.shape[0]):
            t0 = np.square(img_info[:, 0] - cluster_info[i, 0])
            t1 = np.square(img_info[:, 1] - cluster_info[i, 1])
            t2 = np.square(img_info[:, 2] - cluster_info[i, 2])
            # t3 = np.square(img_info[:, 3] - cluster_info[i, 3])
            # Ds[i, :, :] = np.vstack((t0, t1, t2, t3)).transpose()
            Ds[i, :, :] = np.vstack((t0, t1, t2)).transpose()

        del img_info, t0, t1, t2, t3
        self.sum_DS = np.sqrt(Ds.sum(axis=2))
        del Ds
        self.sum_DS = np.argmin(self.sum_DS, axis=0)
        self.sum_DS = self.sum_DS.reshape(self.data.shape[0], self.data.shape[1], self.data.shape[2]).astype('int16')
        print('计算完成')

    def save_current_image(self, name):
        new_mask_img = sitk.GetImageFromArray(self.sum_DS)
        new_mask_img.SetDirection(self.img.GetDirection())
        new_mask_img.SetOrigin(self.img.GetOrigin())
        new_mask_img.SetSpacing(self.img.GetSpacing())
        sitk.WriteImage(new_mask_img, name)


if __name__ == '__main__':
    p = SLICProcessor('lobe512_000_0000.nii.gz', 5, 100,
                      init_clusters_list=[[215, 353, 162], [173, 257, 107], [108, 257, 140], [218, 257, 383],
                                          [109, 257, 399]])
    p.init_clusters()  # 创建种子点
    p.assignment()
    name = 'lobe_5.nii.gz'
    p.save_current_image(name)
    print('图片保存至:{}'.format(name))
