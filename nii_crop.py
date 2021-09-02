import SimpleITK as sitk
import os
import numpy as np
import tqdm


def get_listdir(path):
    tmp_list = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.gz':
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list


def crop(img_path, lobe_mask_path, airway_mask_path, save_path):
    img_sitk = sitk.ReadImage(img_path)
    img_arr = sitk.GetArrayFromImage(img_sitk)
    lobe_mask_sitk = sitk.ReadImage(lobe_mask_path)
    lobe_mask_arr = sitk.GetArrayFromImage(lobe_mask_sitk)
    airway_mask_sitk = sitk.ReadImage(airway_mask_path)
    airway_mask_arr = sitk.GetArrayFromImage(airway_mask_sitk)
    img_arr[lobe_mask_arr != 3] = 0
    lobe_mask_arr[lobe_mask_arr != 3] = 0
    temp = np.zeros_like(airway_mask_arr)
    temp[airway_mask_arr == 6] = 6
    temp[airway_mask_arr == 15] = 15
    temp[airway_mask_arr == 18] = 18
    temp[airway_mask_arr == 19] = 19
    temp[airway_mask_arr == 20] = 20
    airway_mask_arr = temp
    print(img_arr.shape, end=" ")
    for axis in [0, 1, 2]:
        sums = np.sum(np.sum(img_arr, axis=axis), axis=(axis + 1) % 2)

        # Track all =0 layers from front from that axis
        remove_front_index = 0
        while sums[remove_front_index] == 0:
            remove_front_index += 1

        # Track all =0 layers from back from that axis
        remove_back_index = len(sums) - 1
        while sums[remove_back_index] == 0:
            remove_back_index -= 1

        # Remove those layers
        img_arr = np.delete(
            img_arr, list(range(remove_front_index - 1)) + list(range(remove_back_index + 2, len(sums))),
            axis=(axis + 1) % 3
        )
        airway_mask_arr = np.delete(
            airway_mask_arr, list(range(remove_front_index - 1)) + list(range(remove_back_index + 2, len(sums))),
            axis=(axis + 1) % 3
        )
        lobe_mask_arr = np.delete(
            lobe_mask_arr, list(range(remove_front_index - 1)) + list(range(remove_back_index + 2, len(sums))),
            axis=(axis + 1) % 3
        )
        validation_sums = np.sum(np.sum(img_arr, axis=axis), axis=(axis + 1) % 2)
        print(" -> ", img_arr.shape, end=" ")
    img_arr[img_arr == 0] = -1000
    new_mask_img = sitk.GetImageFromArray(img_arr)
    new_mask_img.SetDirection(img_sitk.GetDirection())
    new_mask_img.SetOrigin(img_sitk.GetOrigin())
    new_mask_img.SetSpacing(img_sitk.GetSpacing())
    _, fullflname = os.path.split(img_path)
    sitk.WriteImage(new_mask_img, os.path.join(save_path, 'RL_' + fullflname))

    new_airway_mask_img = sitk.GetImageFromArray(airway_mask_arr)
    new_airway_mask_img.SetDirection(img_sitk.GetDirection())
    new_airway_mask_img.SetOrigin(img_sitk.GetOrigin())
    new_airway_mask_img.SetSpacing(img_sitk.GetSpacing())
    sitk.WriteImage(new_airway_mask_img, os.path.join(save_path, 'RL_airway_' + fullflname))

    new_lobe_mask_img = sitk.GetImageFromArray(lobe_mask_arr)
    new_lobe_mask_img.SetDirection(img_sitk.GetDirection())
    new_lobe_mask_img.SetOrigin(img_sitk.GetOrigin())
    new_lobe_mask_img.SetSpacing(img_sitk.GetSpacing())
    sitk.WriteImage(new_lobe_mask_img, os.path.join(save_path, 'RL_lobe_' + fullflname))


if __name__ == '__main__':
    img_path = r'D:\my_code\segment_registration\my_data\img'
    lobe_mask_path = r'D:\my_code\segment_registration\my_data\mask_lobe'
    airway_mask_path = r'D:\my_code\segment_registration\my_data\mask_airway'
    save_path = r'D:\my_code\segment_registration\my_data'
    img = get_listdir(img_path)
    img.sort()
    l_mask = get_listdir(lobe_mask_path)
    l_mask.sort()
    a_mask = get_listdir(airway_mask_path)
    a_mask.sort()
    for i in tqdm.trange(len(img)):
        crop(img[i], l_mask[i], a_mask[i], save_path)
