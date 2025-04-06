import os
os.chdir('/mnt/f/BaiduSyncdisk/My_projects/My_github/Image_proj')
import go_path as paths
import pandas as pd
import n_tools as nt
import pre_datas
import importlib
import h5pyio
importlib.reload(h5pyio)
# 复制mask_warped.nii.gz到指定目录并筛选相应的患者信息表
if not os.path.exists(paths.patient_info_path):
    patient_info = pre_datas.copy_mask_to_subjects_dir(paths.images_dir, paths.excel_info_path)
else:
    patient_info = pd.read_csv(paths.patient_info_path,dtype='str')

patient_list = patient_info['术前MR文件名'].tolist()

# 复制fsaverage_sym和cvs_avg35_inMNI152到指定目录
nt.copy_temp_sub(paths.temp_sub, paths.sym_sub)

# 读取mask列表
mask_list = [os.path.join(paths.SUBJECTS_DIR,i,'mask_warped.nii.gz') for i in patient_list]

# 批量提取所有患者mask在皮质下的体积
# 将mask转换到cvs152空间
mask_cvs152_list, failed_list = nt.convert_mask_to_cvs152(mask_list, os.path.join(paths.temp_sub, 'mri/aseg.mgz'))
aseg_file = os.path.join(paths.temp_sub, 'mri/aseg.mgz')
lookup_table_file = os.path.join(paths.FREESURFER_HOME, 'FreeSurferColorLUT.txt') #511及后几行有8列的的颜色表需要注释掉

mask_vols, aseg_vols = nt.calc_maskin_aseg(mask_cvs152_list, aseg_file, lookup_table_file)
mask_vols_table = pd.DataFrame([aseg_vols] + mask_vols)
mask_vols_table.to_csv(paths.mask_vols_table_path, index=False)

# 批量提取所有患者mask在皮质表面的体积并转换到fsaverage_sym的左脑表面
surfs_lh, labels_lh = nt.mask_vol2surf(mask_cvs152_list)

# 批量提取所有患者label面积
annot_path = os.path.join(paths.sym_sub, 'label/lh.aparc.annot')
white_surface_path = os.path.join(paths.sym_sub, 'surf/lh.white')

label_surfs, annot_area, annot_names = nt.culc_lesion_area(labels_lh,annot_path,white_surface_path) 
label_surfs_table = pd.DataFrame([annot_area] + label_surfs)
label_surfs_table.to_csv(paths.label_surfs_table_path, index=False)

# 将mask转换为h5py格式
h5pyio.make_lesion_vol_h5py(patient_list, os.path.join(paths.sym_sub, 'mri/orig.mgz'), os.path.join(paths.SUBJECTS_DIR, 'lesion_vol_symsurf.h5py'))
h5pyio.make_lesion_surf_h5py(patient_list, os.path.join(paths.sym_sub, 'surf/lh.white'), os.path.join(paths.SUBJECTS_DIR, 'lesion_surf_symsurf.h5py'))
# 读取h5py文件
h5py_file = os.path.join(paths.SUBJECTS_DIR, 'lesion_surf_symsurf.h5py')
lesion_surf_symsurf= h5pyio.load_h5py_file(h5py_file)
#lesion_surf_symsurf['lesions'][1].sum()
##### test
'''
nt.is_valid_label(os.path.join(SUBJECTS_DIR, 'test', 'surf/mask_in_symsurf_lrholh.label'))
len(labels_area[0])
len(annot_area)
len(annot_names)
annot, ctab, names = fsio.read_annot(annot_path)
len(np.unique(annot))
len(annot)
len(ctab)
np.unique(annot)[5]
names[5]

'''
