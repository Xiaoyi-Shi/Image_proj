import os
import numpy as np
import pandas as pd
import nibabel as nib
import nibabel.freesurfer.io as fsio
import matplotlib.pyplot as plt

def is_valid_label(file_path):
    """
    检查文件是否存在且为有效的 label 文件
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
    
    #读取白质文件
    vertices, faces = fsio.read_geometry(white_surface_file)

    # 读取label and annot文件
    annot, ctab, names = fsio.read_annot(annot_file)
    unique_regions = np.unique(annot[annot >= 0])  # 排除未标记区域 (-1)
    if len(names) != len(unique_regions):
        print("region标签和名字长度不匹配，请检查修改")
        del names[4] # 删除第5个元素，aparc.annot的1004-ctx-lh-corpuscallosum没有在annot里
    region_map_name = {region : names.decode('utf-8') for region, names in zip(unique_regions, names)} 

    triangle_areas = []
    for face in faces:
        v1, v2, v3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        area = compute_triangle_area(v1, v2, v3)
        triangle_areas.append(area)
    triangle_areas = np.array(triangle_areas)

    # 5. 计算每个脑区的总面积
    region_areas = {}
    for region_id in unique_regions:
        # 找到属于该脑区的面片
        region_vertices = np.where(annot == region_id)[0]
        region_faces_mask = np.any(np.isin(faces, region_vertices), axis=1)
        region_area = triangle_areas[region_faces_mask].sum()
        region_areas[region_map_name[region_id]] = region_area

    # 6. 计算自定义 labels 在各个脑区的面积
    fina_stats = []
    for mask_label in mask_label_list:
        region_label_areas = {}
        region_label_areas['patient'] = mask_label
        if not is_valid_label(mask_label):
            region_label_areas['error'] = 'label文件无效'
        else:
            label_vertices = fsio.read_label(mask_label, read_scalars=False)
            label_vertex_set = set(label_vertices)  # 转换为集合以加速查找
            
            for region_id in unique_regions:
                # 找到该脑区与自定义 label 的交集顶点
                region_vertices = np.where(annot == region_id)[0]
                intersection_vertices = label_vertex_set.intersection(region_vertices)
                
                if len(intersection_vertices) > 0:
                    # 找到包含交集顶点的面片
                    intersection_faces_mask = np.any(np.isin(faces, list(intersection_vertices)), axis=1)
                    intersection_area = triangle_areas[intersection_faces_mask].sum()
                    region_label_areas[region_map_name[region_id]] = intersection_area
                else:
                    region_label_areas[region_map_name[region_id]] = 0.0
            fina_stats.append(region_label_areas)
    return fina_stats, region_areas, region_map_name
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

def calc_maskin_aseg(mask_file_list, aseg_file, lookup_table_file):
    """
    计算mask在aseg中的体素数量
    """

    # 加载aseg_file并获取所有唯一区域ID
    aseg_img = nib.load(aseg_file)
    aseg_data = aseg_img.get_fdata()
    unique_regions = np.unique(aseg_data).astype(int)

    # 读取lookup table 
    lookup_df = pd.read_csv(lookup_table_file, 
                           delim_whitespace=True, 
                           comment='#',
                           names=['ID', 'Name', 'R', 'G', 'B', 'A'])
    
    # 计算temp的皮质下体素数量
    temp_aseg_vols = {}
    for region_id in unique_regions:
        if region_id == 0:  # 跳过背景
            continue
        region_mask = aseg_data == region_id
        region_voxel_count = np.sum(region_mask)
        region_name = lookup_df[lookup_df['ID'] == region_id]['Name'].iloc[0]
        temp_aseg_vols[region_name] = region_voxel_count

    # 计算每个mask的体素数量
    fina_stats = []
    for mask_file in mask_file_list:
        results = {}
        results['patient'] = mask_file
        mask_img = nib.load(mask_file)
        mask_data = mask_img.get_fdata()
        mask_binary = (mask_data > 0).astype(int)

        if mask_data.shape != aseg_data.shape:
            results['error'] = 'mask和aparc+aseg的维度不匹配'
        else:
            for region_id in unique_regions:
                if region_id == 0:  # 跳过背景
                    continue
                region_mask = (aseg_data == region_id) & (mask_binary == 1)
                region_voxel_count = np.sum(region_mask)
                region_name = lookup_df[lookup_df['ID'] == region_id]['Name'].iloc[0]
                results[region_name] = region_voxel_count
        fina_stats.append(results)

    return fina_stats, temp_aseg_vols