import copy
import numpy as np
from tqdm import trange
from tqdm import tqdm
import SimpleITK as sitk


class SLICProcessor(object):

    def __init__(self, airway_mask, lobe_mask):
        self.img = sitk.ReadImage(airway_mask)
        self.data = sitk.GetArrayFromImage(self.img).astype('int8')
        self.lobe = sitk.ReadImage(lobe_mask)
        self.lobe_data = sitk.GetArrayFromImage(self.lobe).astype('int8')

    def assignment(self, lobe):
        airway_data = copy.copy(self.data)
        airway_data[self.lobe_data != lobe] = 0
        clusters_list = np.argwhere(airway_data > 1)
        self.value = np.asarray([airway_data[i[0]][i[1]][i[2]] for i in clusters_list])
        clusters_list = np.concatenate((clusters_list, np.expand_dims(self.value, 1)), axis=1)
        lobe_list = np.argwhere(self.lobe_data == lobe)
        for i in tqdm(lobe_list):
            t0 = np.sum(np.square(clusters_list[:, :-1] - i), axis=1)
            self.data[i[0], i[1], i[2]] = clusters_list[np.argmin(t0), 3]

    def save_current_image(self, name):
        self.data[self.lobe_data == 0] = 0
        new_mask_img = sitk.GetImageFromArray(self.data)
        new_mask_img.SetDirection(self.img.GetDirection())
        new_mask_img.SetOrigin(self.img.GetOrigin())
        new_mask_img.SetSpacing(self.img.GetSpacing())
        sitk.WriteImage(new_mask_img, name)


if __name__ == '__main__':
    p = SLICProcessor(airway_mask='./my_data/mask_airway/airway_segments_mask.nii.gz',
                      lobe_mask='./my_data/mask_lobe/lobe_mask.nii.gz')
    p.assignment(1)
    p.assignment(2)
    p.assignment(3)
    p.assignment(4)
    p.assignment(5)
    save_name = 'segments_img.nii.gz'
    p.save_current_image(save_name)
    print('图片保存至:{}'.format(save_name))
