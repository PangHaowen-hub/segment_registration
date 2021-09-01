import SimpleITK as sitk
import os
from tqdm import trange


def get_listdir(path):
    tmp_list = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.gz':
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list


if __name__ == '__main__':
    img_path = r'F:\segment_registration\Registration\original_image\RL_lobe_Similarity'
    save_path = r'F:\segment_registration\Registration\original_image\atlas.nii.gz'

    img_list = get_listdir(img_path)
    img_list.sort()
    atlas_img = sitk.ReadImage(img_list[0])
    atlas_arr = sitk.GetArrayFromImage(atlas_img)
    for i in trange(1, len(img_list)):
        sitk_img = sitk.ReadImage(img_list[i])
        img_arr = sitk.GetArrayFromImage(sitk_img)
        atlas_arr += img_arr
    new_img = sitk.GetImageFromArray(atlas_arr / len(img_list))
    new_img.SetSpacing(atlas_img.GetSpacing())
    new_img.SetOrigin(atlas_img.GetOrigin())
    new_img.SetDirection(atlas_img.GetDirection())
    sitk.WriteImage(new_img, save_path)
