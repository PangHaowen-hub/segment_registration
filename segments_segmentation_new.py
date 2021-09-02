import numpy as np
from tqdm import trange
from tqdm import tqdm
import SimpleITK as sitk


class SLICProcessor(object):

    def __init__(self, filename):
        self.img = sitk.ReadImage(filename)
        self.data = sitk.GetArrayFromImage(self.img).astype('int8')
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
                    t0 = np.sum(np.square(self.clusters_list[:, :-1] - [i, j, k]), axis=1)
                    self.data[i, j, k] = self.clusters_list[np.argmin(t0), 3]

    def save_current_image(self, name):
        new_mask_img = sitk.GetImageFromArray(self.data)
        new_mask_img.SetDirection(self.img.GetDirection())
        new_mask_img.SetOrigin(self.img.GetOrigin())
        new_mask_img.SetSpacing(self.img.GetSpacing())
        sitk.WriteImage(new_mask_img, name)


if __name__ == '__main__':
    img = sitk.ReadImage('./my_data/RL_lobe_img.nii.gz')  # lobe_mask
    data = sitk.GetArrayFromImage(img)
    p = SLICProcessor('./my_data/RL_airway_img.nii.gz')
    p.assignment()
    name = 'RL_segments_img.nii.gz'
    p.data[data == 0] = 0
    p.save_current_image(name)
    print('图片保存至:{}'.format(name))
