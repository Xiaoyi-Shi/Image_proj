import os
import subprocess
import shutil
os.chdir('/mnt/h/sjw/sMRI_eeg/codes')
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

# 
surf_dir = os.path.join(main_dir, 'surf')
if not os.path.exists(surf_dir):
    os.makedirs(surf_dir)
    print(f"创建目录: {surf_dir}")

mask_in_mni = os.path.join(SUBJECTS_DIR, 'mask_warped.nii.gz')

subprocess.run('mri_vol2surf --mov {} --regheader {} --hemi {} --out {} --interp nearest'.format(
    mask_in_mni, 
    temp_sub, 
    lateral, 
    mask_in_mni2surf), shell=True)
