import os
import subprocess
import go_path as paths
import pandas as pd
os.chdir('/mnt/f/BaiduSyncdisk/My_projects/My_github/Image_proj')
import n_tools as nt
# 复制fsaverage_sym和cvs_avg35_inMNI152到指定目录
nt.copy_temp_sub(paths.temp_sub, paths.sym_sub)

# 读取患者列表
patients_list = ['test','test']#pd.read_csv('patients_list.csv')
mask_list = [os.path.join(paths.SUBJECTS_DIR,i,'mask_warped.nii.gz') for i in patients_list]
# 批量提取所有患者mask在皮质下的体积
# 将mask转换到cvs152空间
mask_cvs152_list, failed_list = nt.convert_mask_to_cvs152(mask_list, os.path.join(paths.temp_sub, 'mri/aseg.mgz'))
aseg_file = os.path.join(paths.temp_sub, 'mri/aseg.mgz')
lookup_table_file = os.path.join(paths.FREESURFER_HOME, 'FreeSurferColorLUT.txt') #511及后几行有8列的的颜色表需要注释掉

mask_vols, aseg_vols = nt.calc_maskin_aseg(mask_cvs152_list, aseg_file, lookup_table_file)
mask_vols_table = pd.DataFrame([aseg_vols] + mask_vols)
mask_vols_table.to_csv('mask_vols_table.csv', index=False)

# 批量提取所有患者mask在皮质表面的体积并转换到fsaverage_sym的左脑表面
surfs_lh, labels_lh = nt.mask_vol2surf(mask_cvs152_list)

# 批量提取所有患者label面积
annot_path = os.path.join(paths.sym_sub, 'label/lh.aparc.annot')
white_surface_path = os.path.join(paths.sym_sub, 'surf/lh.white')

labels_area, annot_area, annot_names = nt.culc_lesion_area(labels_lh,annot_path,white_surface_path) 
labels_area_table = pd.DataFrame([annot_area] + labels_area)
labels_area_table.to_csv('labels_area_table.csv', index=False)

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
