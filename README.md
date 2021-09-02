# 使用肺段支气管信息分割肺段
### step1：运行nii_crop.py
输入三个图像：分别为原图、肺叶mask、支气管mask。

输出两个图像：从原图中裁剪的肺叶部分、右肺下叶肺段支气管mask、右肺下叶mask
### step2：运行segmens_segmentation.py
输入两个图像：肺叶mask、支气管mask。

输出一个图像：肺段mask

**unify_shape.py 统一各图像的shape**