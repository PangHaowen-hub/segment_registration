import numpy as np
from tqdm import trange
from tqdm import tqdm
import SimpleITK as sitk
import os


def get_listdir(path):
    tmp_list = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.gz':
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list


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

    def __init__(self, filename, K):
        self.K = K  # 分为5类
        self.img = sitk.ReadImage(filename)
        self.data = sitk.GetArrayFromImage(self.img)
        index_list = np.unique(self.data)
        index0 = np.argwhere(self.data == index_list[0])
        index1 = np.argwhere(self.data == index_list[1])
        index2 = np.argwhere(self.data == index_list[2])
        index3 = np.argwhere(self.data == index_list[3])
        index4 = np.argwhere(self.data == index_list[4])
        self.image_height = self.data.shape[0]
        self.image_width = self.data.shape[1]
        self.image_channel = self.data.shape[2]
        self.clusters = []
        self.dis = np.full((self.image_height, self.image_width, self.image_channel), np.inf)
        self.init_clusters_list = [index0[0], index1[0], index2[0], index3[0], index4[0]]

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
        Ds = np.zeros((cluster_info.shape[0], self.data.shape[0] * self.data.shape[1] * self.data.shape[2], 3),
                      dtype='int32')

        for i in trange(cluster_info.shape[0]):
            t0 = np.square(img_info[:, 0] - cluster_info[i, 0])
            t1 = np.square(img_info[:, 1] - cluster_info[i, 1])
            t2 = np.square(img_info[:, 2] - cluster_info[i, 2])
            Ds[i, :, :] = np.vstack((t0, t1, t2)).transpose()

        del img_info, t0, t1, t2, t3
        self.sum_DS = np.sqrt(Ds.sum(axis=2))
        del Ds
        self.sum_DS = np.argmin(self.sum_DS, axis=0)
        self.sum_DS = self.sum_DS.reshape(self.data.shape[0], self.data.shape[1], self.data.shape[2]).astype('int16')
        self.sum_DS = self.sum_DS + 1
        print('计算完成')

    def save_current_image(self, name):
        new_mask_img = sitk.GetImageFromArray(self.sum_DS)
        new_mask_img.SetDirection(self.img.GetDirection())
        new_mask_img.SetOrigin(self.img.GetOrigin())
        new_mask_img.SetSpacing(self.img.GetSpacing())
        sitk.WriteImage(new_mask_img, name)


if __name__ == '__main__':

    RL_lobe_mask_path = r'F:\segment_registration\Registration\original_image\mask_RL_lobe'
    lobe_mask_list = get_listdir(RL_lobe_mask_path)
    lobe_mask_list.sort()
    RL_bronchi_mask_path = r'F:\segment_registration\Registration\original_image\mask_RL_bronchi'
    RL_bronchi_mask_list = get_listdir(RL_bronchi_mask_path)
    RL_bronchi_mask_list.sort()
    save_path = r'F:\segment_registration\Registration\original_image\segments_segmentation'
    for i in range(30, len(lobe_mask_list)):
        lobe_mask = sitk.ReadImage(lobe_mask_list[i])
        lobe_data = sitk.GetArrayFromImage(lobe_mask)
        p = SLICProcessor(RL_bronchi_mask_list[i], 5)
        p.init_clusters()  # 创建种子点
        p.assignment()
        p.sum_DS[lobe_data == 0] = 0
        _, fullflname = os.path.split(RL_bronchi_mask_list[i])
        p.save_current_image(os.path.join(save_path, fullflname))
        print('图片保存至:{}'.format(os.path.join(save_path, fullflname)))
