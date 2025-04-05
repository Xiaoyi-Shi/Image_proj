import os

# 设置环境变量
FREESURFER_HOME = os.environ.get('FREESURFER_HOME')
SUBJECTS_DIR = '/root/subjects'
os.environ['SUBJECTS_DIR'] = SUBJECTS_DIR

# 设置主要目录
temp_sub = os.path.join(SUBJECTS_DIR, 'cvs_MNI152')
sym_sub = os.path.join(SUBJECTS_DIR, 'fsaverage_sym')

# 获取所有环境变量
env_vars = os.environ

print(f"FREESURFER_HOME: {FREESURFER_HOME}")
print(f"SUBJECTS_DIR: {SUBJECTS_DIR}")
print(f"temp_sub: {temp_sub}")
print(f"sym_sub: {sym_sub}")