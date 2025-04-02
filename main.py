import os
import subprocess
import shutil
import pandas as pd
os.chdir('/mnt/f/BaiduSyncdisk/My_projects/My_github/Image_proj')
import n_tools as nt
# 获取所有环境变量
#env_vars = os.environ
# 设置环境变量
FREESURFER_HOME = os.environ.get('FREESURFER_HOME')
SUBJECTS_DIR = '/root/subjects'
os.environ['SUBJECTS_DIR'] = SUBJECTS_DIR
# 设置主要目录
temp_sub = os.path.join(SUBJECTS_DIR, 'cvs_MNI152')
sym_sub = os.path.join(SUBJECTS_DIR, 'fsaverage_sym')
# 复制模板
if not os.path.exists(temp_sub):
    shutil.copytree(os.path.join(FREESURFER_HOME,'subjects/cvs_avg35_inMNI152'), temp_sub)
if not os.path.exists(sym_sub):
    shutil.copytree(os.path.join(FREESURFER_HOME,'subjects/fsaverage_sym'), sym_sub)
# 制作对称模版
subprocess.run('surfreg --s {} --t {} --lh --no-annot'.format(
    os.path.basename(temp_sub), 
    os.path.basename(sym_sub)), shell=True)
subprocess.run('surfreg --s {} --t {} --lh --xhemi --no-annot'.format(
    os.path.basename(temp_sub), 
    os.path.basename(sym_sub)), shell=True)

# 读取患者列表并批量将mask_vol2surf
patients_list = ['test','test']#pd.read_csv('patients_list.csv')
patient_dir = os.path.join(SUBJECTS_DIR, 'test')
mask_in_mni = os.path.join(patient_dir, 'mask_warped.nii.gz')
lateral = nt.determine_mask_side(mask_in_mni)
surf_dir = os.path.join(patient_dir, 'surf')
if not os.path.exists(surf_dir):
    os.makedirs(surf_dir)
    print(f"创建目录: {surf_dir}")
mask_in_surf = os.path.join(patient_dir, 'surf/mask_in_surf_'+lateral+'.mgh')

subprocess.run('mri_vol2surf --mov {} --regheader {} --hemi {} --out {} --interp nearest'.format(
    mask_in_mni, 
    os.path.basename(temp_sub), 
    lateral, 
    mask_in_surf), shell=True)

# 进行对称化并将mask_in_surf转换为fsaverage_sym的lh
if lateral == 'lh':
    mask_in_symsurf_lh = os.path.join(patient_dir, 'surf/mask_in_symsurf_lolh.mgh')
    subprocess.run('mris_apply_reg --src {} --trg {} --streg {} {}'.format(
        mask_in_surf,
        mask_in_symsurf_lh,
        os.path.join(temp_sub, 'surf/lh.sphere.reg'),
        os.path.join(sym_sub, 'surf/lh.sphere.reg')
    ), shell=True)
if lateral == 'rh':
    mask_in_symsurf_lh = os.path.join(patient_dir, 'surf/mask_in_symsurf_rolh.mgh')
    subprocess.run('mris_apply_reg --src {} --trg {} --streg {} {}'.format(
        mask_in_surf,
        mask_in_symsurf_lh,
        os.path.join(temp_sub, 'xhemi/surf/lh.fsaverage_sym.sphere.reg'),
        os.path.join(sym_sub, 'surf/lh.sphere.reg')
    ), shell=True)

# surf文件转化为label
label_in_symsurf_lh = os.path.join(patient_dir, 'surf/mask_in_symsurf_lrholh.label')
subprocess.run('mri_cor2label --i {} --id {} --l {} --surf {} {} --remove-holes-islands'.format(
    mask_in_symsurf_lh,
    '1',
    label_in_symsurf_lh,
    os.path.basename(sym_sub),
    'lh'), shell=True)

# 批量提取所有患者label面积
labels_list = [os.path.join(SUBJECTS_DIR,i,'surf/mask_in_symsurf_lrholh.label') for i in patients_list]
annot_path = os.path.join(sym_sub, 'label/lh.aparc.annot')
white_surface_path = os.path.join(sym_sub, 'surf/lh.white')

labels_area, annot_area, annot_names = nt.culc_lesion_area(labels_list,annot_path,white_surface_path) 
labels_area_table = pd.DataFrame([annot_area] + labels_area)
labels_area_table.to_csv('labels_area_table.csv', index=False)

# 批量提取所有患者mask在皮质下的体积
subprocess.run('mri_convert {} {} --like {} -rt nearest'.format(
    os.path.join(SUBJECTS_DIR, 'test/mask_warped.nii.gz'),
    os.path.join(SUBJECTS_DIR, 'test/mask_warped_converted.nii.gz'),
    os.path.join(temp_sub, 'mri/aseg.mgz')), shell=True)
mask_list = [os.path.join(SUBJECTS_DIR,i,'mask_warped_converted.nii.gz') for i in patients_list]
aseg_file = '/root/subjects/cvs_MNI152/mri/aparc+aseg.mgz'
lookup_table_file = '/mnt/h/djh/nilearn_projects/fsaverage_sym/FreeSurferColorLUT.txt'

mask_vols, aseg_vols = nt.calc_maskin_aseg(mask_list, aseg_file, lookup_table_file)
mask_vols_table = pd.DataFrame([aseg_vols] + mask_vols)
mask_vols_table.to_csv('mask_vols_table.csv', index=False)
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
