import os
import pandas as pd
import go_path as paths
import n_tools as nt
import numpy as np
import nilearn.surface as nis
import nibabel as nib
from tqdm import tqdm

def copy_mask_to_subjects_dir(images_dir, excel_info_path):
    """
    将mask_warped.nii.gz文件复制到指定的subjects目录下,并筛选相应的患者信息表
    ---------
    - images_dir: str
        原始数据目录，包含患者mask的文件夹
    - excel_info_path: str
        患者信息表路径，包含患者的MR文件名
    ---------
    returns:
    - patient_info_real: DataFrame
        筛选后的患者信息表
    - patient_list_real: list
        筛选后的患者列表
    """

    excel_info = pd.read_excel(excel_info_path,dtype='str')
    excel_info = excel_info.replace(r'\s+', '', regex=True) #去除所有空格
    patient_list_excel = excel_info['检查流水号'].tolist()
    patient_list_real = []
    for i in os.scandir(images_dir):
        IDs = i.name.split("_seg_mask")[0]
        if i.name.endswith(".nii.gz")  and IDs in patient_list_excel:
            segmentation = i.path
            #images_dir = os.path.join(i.path,'atlas/registration')
            if not os.path.exists(os.path.join(paths.SUBJECTS_DIR, IDs)):
                os.makedirs(os.path.join(paths.SUBJECTS_DIR, IDs))
            mask = nib.load(segmentation)
            mask_data = mask.get_fdata().astype(np.float32)
            header = mask.header
            header['scl_slope'] = 1.0
            header['scl_inter'] = 0.0
            fixed_file = os.path.join(paths.SUBJECTS_DIR, IDs, 'seg_mask.nii.gz')
            new_img = nib.Nifti1Image(mask_data, mask.affine, header)
            nib.save(new_img, fixed_file)
            #os.system(f"cp -f {segmentation} {os.path.join(paths.SUBJECTS_DIR, IDs, 'seg_mask.nii.gz')}")
            patient_list_real.append(IDs)
    patient_info_real = excel_info[excel_info['检查流水号'].isin(patient_list_real)]

    patient_info_real.to_csv(paths.patient_info_path, index=False)
    return patient_info_real

def lesion_relative_clinvariable(lesions_array, clin_data, feature_names, lesion_type):
    """
    计算每个病灶在临床变量上的平均值
    ----------
    - lesions_array: np.ndarray
        病灶数组，形状为(n_patients, n_lesions)
    - clin_data: DataFrame
        临床数据，包含患者的临床变量
    - feature_names: list of str
        临床变量的名称
    - lesion_type: str
        病灶类型，'surf'或'vol'
    ----------
    returns:
    - None
    """
    
    lesions_array = lesions_array.astype(bool)
    demo_features = clin_data[feature_names].astype(np.float32).interpolate().to_numpy()

    for feature_name in feature_names:
        mask_means = np.zeros(lesions_array.shape[1])
        for vertex in tqdm(np.arange(lesions_array.shape[1]), desc='Calculating means for each lesion mask'):
            if np.sum(lesions_array[:,vertex])>3:
                mask_means[vertex]=np.mean(demo_features[:,list(feature_names).index(feature_name)][lesions_array[:,vertex]])
            #else:
            #    mask_means[vertex]=1

        if lesion_type == 'surf':

            surf_result_dir = os.path.join(paths.states_results_dir,'surf')
            if not os.path.exists(surf_result_dir):
                os.makedirs(surf_result_dir)

            mesh = nis.PolyMesh(
                left = nis.load_surf_mesh(os.path.join(paths.sym_sub,'surf/lh.inflated'))
            )
            data = nis.PolyData( left= mask_means)
            mesh.to_filename(os.path.join(surf_result_dir, 'lh_inflated_mesh.gii'))
            data.to_filename(os.path.join(surf_result_dir, f'lh_{feature_name}_data.gii'))

        elif lesion_type == 'vol' and lesions_array.shape[1] == 256*256*256:

            vol_result_dir = os.path.join(paths.states_results_dir,'mri')
            if not os.path.exists(vol_result_dir):
                os.makedirs(vol_result_dir)

            mask_means = mask_means.reshape(256,256,256)
            template_nii = nib.load(os.path.join(paths.temp_sub, 'mri/orig.mgz'))
            output_mask_res = nib.Nifti1Image(mask_means,
                                        affine = template_nii.affine,
                                        header = template_nii.header
                                        )
            nib.save(output_mask_res,os.path.join(vol_result_dir, f'{feature_name}_data.mgz'))

        else:
            print('Invalid lesion type or shape of lesions_array')
            return None
    return None

if __name__ == '__main__':
    print("test")