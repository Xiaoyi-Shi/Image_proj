import os
import numpy as np
import nibabel as nib
import nibabel.freesurfer.io as fsio
import matplotlib.pyplot as plt

def is_valid_label(file_path):
    """
    检查文件是否存在且为有效的 NIfTI 文件
    """
    if not os.path.exists(file_path):
        return False
    try:
        fsio.read_label(file_path, read_scalars=False)
        return True
    except Exception as e:
        return False
    
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

def compute_triangle_area(v1, v2, v3):
    # 使用叉积计算三角形面积
    a = v2 - v1
    b = v3 - v1
    cross_product = np.cross(a, b)
    area = 0.5 * np.linalg.norm(cross_product)
    return area

def culc_lesion_area(mask_label_list, annot_file, white_surface_file):
    # 读取label and annot文件
    annot, ctab, names = fsio.read_annot(annot_file)
    vertices, faces = fsio.read_geometry(white_surface_file)

    triangle_areas = []
    for face in faces:
        v1, v2, v3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        area = compute_triangle_area(v1, v2, v3)
        triangle_areas.append(area)
    triangle_areas = np.array(triangle_areas)

    # 5. 计算每个脑区的总面积
    unique_regions = np.unique(annot[annot >= 0])  # 排除未标记区域 (-1)
    region_areas = {}
    for region_id in unique_regions:
        # 找到属于该脑区的面片
        region_vertices = np.where(annot == region_id)[0]
        region_faces_mask = np.any(np.isin(faces, region_vertices), axis=1)
        region_area = triangle_areas[region_faces_mask].sum()
        region_areas[region_id] = region_area

    # 6. 计算自定义 labels 在各个脑区的面积
    fina_stats = []
    for mask_label in mask_label_list:
        label_vertices = fsio.read_label(mask_label, read_scalars=False)
        label_vertex_set = set(label_vertices)  # 转换为集合以加速查找
        region_label_areas = {}
        for region_id in unique_regions:
            # 找到该脑区与自定义 label 的交集顶点
            region_vertices = np.where(annot == region_id)[0]
            intersection_vertices = label_vertex_set.intersection(region_vertices)
            
            if len(intersection_vertices) > 0:
                # 找到包含交集顶点的面片
                intersection_faces_mask = np.any(np.isin(faces, list(intersection_vertices)), axis=1)
                intersection_area = triangle_areas[intersection_faces_mask].sum()
                region_label_areas[region_id] = intersection_area
            else:
                region_label_areas[region_id] = 0.0
        fina_stats.append(region_label_areas)
    return fina_stats, region_areas, names
    """# 7. 输出结果
    for region_id, area in region_label_areas.items():
        region_name = names[region_id].decode('utf-8')  # 转换为字符串
        total_area = region_areas[region_id]
        if total_area > 0:
            proportion = area / total_area * 100
            print(f"{region_name}: {area:.2f} mm² (占该脑区 {proportion:.2f}%)")
        else:
            #print(f"{region_name}: {area:.2f} mm² (该脑区总面积为 0)")
            pass
    """
    