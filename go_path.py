import os

# 设置环境变量
FREESURFER_HOME = os.environ.get('FREESURFER_HOME')
SUBJECTS_DIR = '/root/subjects'
os.environ['SUBJECTS_DIR'] = SUBJECTS_DIR

# 设置主要目录
temp_sub = os.path.join(SUBJECTS_DIR, 'cvs_MNI152')
sym_sub = os.path.join(SUBJECTS_DIR, 'fsaverage_sym')
# 设置数据目录
images_dir = '/mnt/h/sjw/sMRI_eeg/glioma_MR_eeg_ants_163/'
excel_info_path = '/mnt/h/sjw/sMRI_eeg/MR_patient_info.xlsx'

# 设置输出目录
patient_info_path = os.path.join(SUBJECTS_DIR, 'patients_info.csv')
mask_vols_table_path = os.path.join(SUBJECTS_DIR, 'mask_vols_table.csv')
label_surfs_table_path = os.path.join(SUBJECTS_DIR, 'label_surfs_table.csv')
# 获取所有环境变量
env_vars = os.environ

print(f"FREESURFER_HOME: {FREESURFER_HOME}")
print(f"SUBJECTS_DIR: {SUBJECTS_DIR}")
print(f"temp_sub: {temp_sub}")
print(f"sym_sub: {sym_sub}")