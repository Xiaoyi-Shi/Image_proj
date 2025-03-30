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
patients_list = pd.read_csv('patients_list.csv')
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
    subprocess.run('mris_apply_reg --src {} --trg {} --streg {} {}'.format(
        mask_in_surf,
        os.path.join(patient_dir, 'surf/mask_in_symsurf_lolh.mgh'),
        os.path.join(temp_sub, 'surf/lh.sphere.reg'),
        os.path.join(sym_sub, 'surf/lh.sphere.reg')
    ), shell=True)
if lateral == 'rh':
    subprocess.run('mris_apply_reg --src {} --trg {} --streg {} {}'.format(
        mask_in_surf,
        os.path.join(patient_dir, 'surf/mask_in_symsurf_rolh.mgh'),
        os.path.join(temp_sub, 'xhemi/surf/lh.fsaverage_sym.sphere.reg'),
        os.path.join(sym_sub, 'surf/lh.sphere.reg')
    ), shell=True)
