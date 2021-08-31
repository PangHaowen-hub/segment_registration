import cv2
import imutils
import SimpleITK as sitk
import numpy as np


# 骨架图依赖灰度图
#
# img = sitk.ReadImage('airway_mask_000.nii.gz')
# data = sitk.GetArrayFromImage(img)


gray = cv2.imread('airway_mask_000.nii.png', 0)  # 直接读取为灰度图
cv2.imshow("gray", gray)

# 骨架化图像
skeleton = imutils.skeletonize(gray, size=(3, 3))
cv2.imshow("Skeleton", skeleton)
cv2.waitKey(0)
