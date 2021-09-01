import os
import glob
import ants
import numpy as np
import SimpleITK as sitk
from tqdm import trange
'''
ants.registration()函数的返回值是一个字典：
    warpedmovout: 配准到fixed图像后的moving图像 
    warpedfixout: 配准到moving图像后的fixed图像 
    fwdtransforms: 从moving到fixed的形变场 
    invtransforms: 从fixed到moving的形变场

type_of_transform参数的取值可以为：
    Translation:平移变换
    Rigid:刚性变换:仅旋转和平移。
    Similarity:相似变换:缩放、旋转和平移。
    QuickRigid
    DenseRigid
    BOLDRigid
    Affine:仿射变换:刚性+缩放。
    AffineFast
    BOLDAffine
    TRSAA:translation, rigid, similarity, affine (twice), please set regIterations if using this option. 
    Elastic:弹性变形:仿射+变形。
    ElasticSyN:对称归一化:仿射+变形变换，以互信息为优化准则，以elastic为正则项。
    SyN:对称归一化:仿射配准+可变形配准，以互信息为优化准则
    SyNRA:对称归一化:刚性+仿射+变形，互信息为优化度量。
    SyNOnly:对称归一化:不进行初始变换，以互信息为优化度量。假设图像已对齐。如果你想运行一个非掩码仿射，然后是掩码可变形配准。
    SyNCC:SyN，但用互相关作为度量。
    SyNabp
    SyNBold
    SyNBoldAff
    SyNAggro:效果更好的SyN，用时比SyN长。 
    TVMSQ:具有均方度量的时变微分同胚
'''


def get_listdir(path):
    tmp_list = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.gz':
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list


def ants_reg(f_img_path, m_img_path):
    f_img = ants.image_read(f_img_path)
    m_img = ants.image_read(m_img_path)

    mytx = ants.registration(fixed=f_img, moving=m_img, type_of_transform='SyN')
    # mytx = ants.registration(fixed=f_img, moving=m_img, type_of_transform='Similarity')
    # 将形变场作用于moving图像，得到配准后的图像
    warped_img = ants.apply_transforms(fixed=f_img, moving=m_img, transformlist=mytx['fwdtransforms'],
                                       interpolator="linear")

    warped_img.set_direction(f_img.direction)
    warped_img.set_origin(f_img.origin)
    warped_img.set_spacing(f_img.spacing)
    save_path = r"./data/RL_lobe_syn"
    _, img_fullflname = os.path.split(m_img_path)
    ants.image_write(warped_img, os.path.join(save_path, img_fullflname))
    print(os.path.join(save_path, img_fullflname))


if __name__ == '__main__':
    f_img = r'./data/RL_lobe_norm/lobe512_000_0000.nii.gz'
    m_img_list = get_listdir(r'./data/RL_lobe_norm')
    m_img_list.sort()
    for i in trange(len(m_img_list)):
        ants_reg(f_img, m_img_list[i])
