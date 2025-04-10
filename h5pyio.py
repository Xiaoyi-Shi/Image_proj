import os
import numpy as np
import h5py
import n_tools as nt
import nibabel as nib
import nibabel.freesurfer.io as fsio
import go_path as paths

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

def make_lesion_vol_h5py(patient_list, ref_orig ,h5py_file):
    """
    """
    mask_array = []
    patients_list_valid = []
    falied_list = []
    orig_data = nib.load(ref_orig).get_fdata()
    image_sahpe = orig_data.shape
    for patient in patient_list:
        mask = os.path.join(paths.SUBJECTS_DIR, patient, 'seg_mask_cvs152.nii.gz')
        if nt.is_valid_vol(mask):
            mask_data = nib.load(mask).get_fdata().astype(bool)
            if mask_data.shape == image_sahpe:
                mask_data_flatten = mask_data.flatten()
                mask_array.append(mask_data_flatten)
                patients_list_valid.append(patient)
            else:
                print(f'维度不匹配：{patient}')
                falied_list.append(patient)
        else:
            print(f"文件不存在或无法打开: {patient}")
            falied_list.append(patient)
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

def make_lesion_surf_h5py(patient_list, ref_white ,h5py_file):
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
    for patient in patient_list:
        #lesion = os.path.join(paths.SUBJECTS_DIR, patient,'surf/mask_in_symsurf_lrholh.label') #调试
        lesion = os.path.join(paths.SUBJECTS_DIR, patient,'surf/mask_in_symsurf_lrholh.label')
        if nt.is_valid_label(lesion):
            label_index = fsio.read_label(lesion, read_scalars=False)
            label_array = np.zeros(vert_num)
            label_array[label_index] = 1
            lesions_array.append(label_array)
            patients_list_valid.append(patient)
        else:
            print(f"文件不存在或无法打开: {patient}")
            falied_list.append(patient)
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
    # 测试读取h5py文件
    h5py_file = os.path.join(paths.SUBJECTS_DIR, 'lesion_surf_symsurf.h5py')
    lesion_surf_symsurf= load_h5py_file(h5py_file)
    print(lesion_surf_symsurf.keys())
    print(lesion_surf_symsurf['lesions'].shape)
    print(lesion_surf_symsurf['patients_list'])