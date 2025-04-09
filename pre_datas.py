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
        原始数据目录，包含患者文件夹
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
        if i.is_dir() and i.name in patient_list_excel:
            segmentation = os.path.join(i.path,'atlas/segmentation',i.name+'_seg_mask.nii.gz')
            images_dir = os.path.join(i.path,'atlas/registration')
            if os.path.exists(segmentation):
                if not os.path.exists(os.path.join(paths.SUBJECTS_DIR, i.name)):
                        os.makedirs(os.path.join(paths.SUBJECTS_DIR, i.name))
                maskfile = nib.load(segmentation)
                maskdata = maskfile.get_fdata()
                #maskdata = maskdata.astype(np.bool).astype(np.double)
                header = maskfile.header
                header['scl_slope'] = 1.0
                header['scl_inter'] = 0.0
                fixed_file = os.path.join(paths.SUBJECTS_DIR, i.name, 'seg_mask.nii.gz')
                new_img = nib.Nifti1Image(maskdata, maskfile.affine, header)
                nib.save(new_img, fixed_file)
                #os.system(f'cp -f {segmentation} {os.path.join(paths.SUBJECTS_DIR, i.name, 'seg_mask.nii.gz')}')
                patient_list_real.append(i.name)
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

    import nilearn.plotting as nip
    import nilearn.surface as nis
    import h5pyio

    h5py_file = os.path.join(paths.states_results_dir, 'lesion_surf_symsurf.h5py')
    lesion_surf_symsurf= h5pyio.load_h5py_file(h5py_file)
    lesions_array = lesion_surf_symsurf['lesions']
    clin_data = pd.read_csv(paths.patient_info_path, dtype='str')
    clin_data.columns
    feature_names = ['年龄','术后癫痫']
    feature_name = '术后癫痫'

    mesh = nis.PolyMesh(
        left = nis.load_surf_mesh(os.path.join(paths.sym_sub,'surf/lh.pial')),
    )
    data = nis.PolyData(left= mask_means)
    new_mesh = nis.SurfaceImage(mesh=mesh, data=data)
    #lesionss = np.sum(lesions_array, axis=0, keepdims=True)
    data = nis.PolyData( left= lesionss.reshape(-1,))
    new_mesh = nis.SurfaceImage(mesh=mesh, data=data)
    fig = nip.view_surf(surf_map = new_mesh,hemi='left',
                            colorbar=True,
                            threshold=0,
                            symmetric_cmap = False,
                            cmap='jet',
                            bg_map=os.path.join(paths.sym_sub,'surf/lh.sulc'))
    fig.open_in_browser()

    #从vol重建surf
    h5py_file = os.path.join(paths.states_results_dir, 'lesion_vol_symsurf.h5py')
    lesion_surf_symsurf= h5pyio.load_h5py_file(h5py_file)
    lesions_array = lesion_surf_symsurf['lesions']
    lesions_sum = np.sum(lesions_array, axis=0, keepdims=True)
    lesions_sum = lesions_sum.reshape(256,256,256)
    template_nii = nib.load(os.path.join(paths.temp_sub, 'mri/orig.mgz'))
    output_mask_res = nib.Nifti1Image(lesions_sum,
                                affine = template_nii.affine,
                                header = template_nii.header
                                )
    surf_mesh = nis.PolyMesh(
        left = nis.load_surf_mesh(os.path.join(paths.temp_sub,'surf/lh.pial')),
        right = nis.load_surf_mesh(os.path.join(paths.temp_sub,'surf/rh.pial'))
    )
    inner_mesh = nis.PolyMesh(
        left = nis.load_surf_mesh(os.path.join(paths.temp_sub,'surf/lh.white')),
        right = nis.load_surf_mesh(os.path.join(paths.temp_sub,'surf/rh.white'))
    )
    vol_in_surf_l = nis.vol_to_surf(output_mask_res, surf_mesh = os.path.join(paths.temp_sub,'surf/lh.pial'),
                                  inner_mesh = os.path.join(paths.temp_sub,'surf/lh.white'),
                                  radius = 1,
                                  interpolation = 'linear',
                                  kind = 'depth',
                                  depth = [10,5])
    vol_in_surf_r = nis.vol_to_surf(output_mask_res, surf_mesh = os.path.join(paths.temp_sub,'surf/rh.pial'),
                                  inner_mesh = os.path.join(paths.temp_sub,'surf/rh.white'),
                                  radius = 1,
                                  interpolation = 'linear',
                                  kind = 'depth',
                                  depth = [10,5])
    
    data = nis.PolyData( left= vol_in_surf_l, right = vol_in_surf_r)
    new_mesh = nis.SurfaceImage(mesh=inner_mesh, data=data)
    fig = nip.view_surf(surf_map = new_mesh,hemi = 'right',
                            colorbar=True,
                            threshold=0,
                            symmetric_cmap = False,
                            cmap='jet',
                            #bg_map=os.path.join(paths.temp_sub,'surf/rh.sulc')
                        )
    fig.open_in_browser()

    