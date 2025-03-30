import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def determine_mask_side(mask_path):
    # 加载 mask 文件
    img = nib.load(mask_path)
    data = img.get_fdata()  # 获取体视素数据
    affine = img.affine     # 获取仿射矩阵

    # 找到掩膜中非零值的坐标
    mask_coords = np.where(data > 0)  # 返回 (x, y, z) 坐标的元组
    if len(mask_coords[0]) == 0:
        return "掩膜为空，无法判断"

    # 计算掩膜的重心（平均坐标）
    centroid = np.mean(mask_coords, axis=1)  # [x_mean, y_mean, z_mean]

    # 将体视素坐标转换为世界坐标（可选，取决于需求）
    world_centroid = nib.affines.apply_affine(affine, centroid)

    # 在 RAS+ 坐标系中，X > 0 表示右脑，X < 0 表示左脑
    x_coord = world_centroid[0]
    if x_coord > 0:
        return "rh"
    elif x_coord < 0:
        return "lh"
    else:
        return "unknown"
