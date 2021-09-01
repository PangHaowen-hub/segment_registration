import cv2
import numpy as np
from PIL import Image
import SimpleITK as sitk
from scipy import ndimage

img = 'airway_segments_mask.nii.gz'
sitk_img = sitk.ReadImage(img)
img_arr = sitk.GetArrayFromImage(sitk_img)
x = img_arr.astype(np.uint8)
# kernel = np.ones((20, 20), np.uint8)
# dilation = cv2.dilate(x, kernel)
for i in range(5):
    x = ndimage.grey_dilation(x, size=(3, 3, 3))

new_img = sitk.GetImageFromArray(x)
new_img.SetSpacing(sitk_img.GetSpacing())
new_img.SetOrigin(sitk_img.GetOrigin())
new_img.SetDirection(sitk_img.GetDirection())
sitk.WriteImage(new_img, 'airway_segments_mask_erosion2.nii.gz')