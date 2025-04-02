import os
import numpy as np
import h5py
import n_tools as nt
import nibabel as nib
import nibabel.freesurfer.io as fsio

def load_h5py_file(file_path):
    """
    读取h5py文件
    ----------
    - file_path: str, h5py文件路径
    ----------
    returns:
    - data: dict, 包含所有数据集的字典
    """
    with h5py.File(file_path, 'r') as f:
        data = {key: f[key][()] for key in f.keys()}
    return data

def make_lesion_vol_h5py(mask_list, ref_orig ,h5py_file):
    """
    """
    mask_array = []
    patients_list_valid = []
    falied_list = []
    orig_data = nib.load(ref_orig).get_fdata()
    image_sahpe = orig_data.shape
    for mask in mask_list:
        if nt.is_valid_vol(mask):
            mask_data = nib.load(mask).get_fdata().astype(bool)
            if mask_data.shape == image_sahpe:
                mask_data_flatten = mask_data.flatten()
                mask_array.append(mask_data_flatten)
                patients_list_valid.append(mask)
            else:
                print(f'维度不匹配：{mask}')
                falied_list.append(mask)
        else:
            print(f"文件不存在或无法打开: {mask}")
            falied_list.append(mask)
    print(f"体素数：{image_sahpe}")
    print(f"有效的mask文件数量: {len(patients_list_valid)}")
    print(f"无效的mask文件数量: {len(falied_list)}")
    for f in falied_list:
        print(f"无效的mask文件: {f}")
    mask_array = np.array(mask_array)

    with h5py.File(h5py_file, 'w') as f:
        f.create_dataset('lesions', data=mask_array)
        f.create_dataset('patients_list', data=np.array(patients_list_valid, dtype='S'))
        print(f'文件保存于：{h5py_file}')

def make_lesion_surf_h5py(lesions_list, ref_white ,h5py_file):
    """
    将surf_overlay的label文件转换为h5py格式
    ----------
    - lesions_list: list of str, 自定义label文件路径列表
    - h5py_file: str, 输出的h5py文件路径
    ----------
    returns:
    - None
    """
    lesions_array = []
    patients_list_valid = []
    falied_list = []
    vertices, faces = nib.freesurfer.read_geometry(ref_white)
    vert_num = len(vertices)
    for lesion in lesions_list:
        if nt.is_valid_label(lesion):
            label_index = fsio.read_label(lesion, read_scalars=False)
            label_array = np.zeros(vert_num)
            label_array[label_index] = 1
            lesions_array.append(label_array)
            patients_list_valid.append(lesion)
        else:
            print(f"文件不存在或无法打开: {lesion}")
            falied_list.append(lesion)
    print(f"面积节点数：{vert_num}")
    print(f"有效的label文件数量: {len(patients_list_valid)}")
    print(f"无效的label文件数量: {len(falied_list)}")
    for f in falied_list:
        print(f"无效的label文件: {f}")
    lesions_array = np.array(lesions_array)

    with h5py.File(h5py_file, 'w') as f:
        f.create_dataset('vertices', data=vertices)
        f.create_dataset('faces', data=faces)
        f.create_dataset('lesions', data=lesions_array)
        f.create_dataset('patients_list', data=np.array(patients_list_valid, dtype='S'))
        print(f'文件保存于：{h5py_file}')

if __name__ == "__main__":
    # 示例用法
    lesions_list = ['/root/subjects/test/surf/mask_in_symsurf_lrholh.label','/root/subjects/test/surf/mask_in_symsurf_lrholh.label']
    ref_white = '/root/subjects/fsaverage_sym/surf/lh.white'
    h5py_file = 'mask_in_symsurf_lrholh.h5py'
    make_lesion_surf_h5py(lesions_list, ref_white, h5py_file)

    # 读取h5py文件
    data = load_h5py_file(h5py_file)

    ###
    mask_list = ['/root/subjects/test/mask_warped_converted.nii.gz','/root/subjects/test/mask_warped_converted.nii.gz']
    ref_orig = '/root/subjects/cvs_MNI152/mri/orig.mgz'
    h5py_file = 'mask_in_orig.h5py'
    make_lesion_vol_h5py(mask_list, ref_orig, h5py_file)

    data = load_h5py_file(h5py_file)
    data['lesions']