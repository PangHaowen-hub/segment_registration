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


class SLICProcessor(object):

    def __init__(self, filename, lobe_name):
        self.img = sitk.ReadImage(filename)
        self.data = sitk.GetArrayFromImage(self.img).astype('int8')

        self.lobe_mask = sitk.ReadImage(lobe_name)
        self.lobe_data = sitk.GetArrayFromImage(self.lobe_mask)

        self.image_height = self.data.shape[0]
        self.image_width = self.data.shape[1]
        self.image_channel = self.data.shape[2]
        self.clusters = []
        self.dis = np.full((self.image_height, self.image_width, self.image_channel), np.inf)
        self.clusters_list = np.argwhere(self.data != 0)
        self.value = np.asarray([self.data[i[0]][i[1]][i[2]] for i in self.clusters_list])
        self.clusters_list = np.concatenate((self.clusters_list, np.expand_dims(self.value, 1)), axis=1)

    def assignment(self):
        for i in trange(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                for k in range(self.data.shape[2]):
                    if self.lobe_data[i, j, k] != 0:
                        t0 = np.sum(np.square(self.clusters_list[:, :-1] - [i, j, k]), axis=1)
                        self.data[i, j, k] = self.clusters_list[np.argmin(t0), 3]

    def save_current_image(self, name):
        self.data[self.lobe_data == 0] = 0
        new_mask_img = sitk.GetImageFromArray(self.data)
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
    save_path = r'F:\segment_registration\Registration\original_image\segments_segmentation_new'
    for i in range(60, len(lobe_mask_list)):
        p = SLICProcessor(RL_bronchi_mask_list[i], lobe_mask_list[i])
        p.assignment()
        _, fullflname = os.path.split(RL_bronchi_mask_list[i])
        p.save_current_image(os.path.join(save_path, fullflname))
        print('图片保存至:{}'.format(os.path.join(save_path, fullflname)))