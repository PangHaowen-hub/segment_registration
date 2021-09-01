import numpy as np
import SimpleITK as sitk
import os
import tqdm


def get_listdir(path):
    tmp_list = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.gz':
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list


def u_shape(img_path, shape, save_path):
    img_sitk = sitk.ReadImage(img_path)
    img_arr = sitk.GetArrayFromImage(img_sitk)
    print(shape, '=>', img_arr.shape)
    x0 = (shape[0] - img_arr.shape[0]) // 2
    x1 = shape[0] - img_arr.shape[0] - x0

    y0 = (shape[1] - img_arr.shape[1]) // 2
    y1 = shape[1] - img_arr.shape[1] - y0

    z0 = (shape[2] - img_arr.shape[2]) // 2
    z1 = shape[2] - img_arr.shape[2] - z0

    new_model = np.ones(shape) * -1000
    new_model[x0:-x1, y0:-y1, z0:-z1] = img_arr

    new_mask_img = sitk.GetImageFromArray(new_model)
    new_mask_img.SetSpacing(img_sitk.GetSpacing())
    new_mask_img.SetOrigin(img_sitk.GetOrigin())
    new_mask_img.SetDirection(img_sitk.GetDirection())
    _, fullflname = os.path.split(img_path)
    sitk.WriteImage(new_mask_img, os.path.join(save_path, fullflname))


if __name__ == '__main__':
    shape = [352, 352, 256]
    img_path = r'F:\segment_registration\Registration\original_image\RL_lobe'
    save_path = r'F:\segment_registration\Registration\original_image\RL_lobe_unify'
    l_img = get_listdir(img_path)
    l_img.sort()
    for i in tqdm.trange(len(l_img)):
        u_shape(l_img[i], shape, save_path)
