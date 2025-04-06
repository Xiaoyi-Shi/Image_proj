import os
import pandas as pd
import go_path as paths
import n_tools as nt

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
    patient_list_excel = excel_info['术前MR文件名'].tolist()
    patient_list_real = []
    for i in os.scandir(images_dir):
        if i.is_dir() and i.name in patient_list_excel:
            for j in os.scandir(i.path):
                if j.name == 'mask_warped.nii.gz'  and nt.is_valid_vol(j.path):
                    if not os.path.exists(os.path.join(paths.SUBJECTS_DIR, i.name)):
                        os.makedirs(os.path.join(paths.SUBJECTS_DIR, i.name))
                    # 复制mask_warped.nii.gz到指定目录
                    os.system(f'cp -f {j.path} {os.path.join(paths.SUBJECTS_DIR, i.name)}')
                    patient_list_real.append(i.name)
                    break
    patient_info_real = excel_info[excel_info['术前MR文件名'].isin(patient_list_real)]

    patient_info_real.to_csv(os.path.join(paths.SUBJECTS_DIR, 'patients_info.csv'), index=False)
    return patient_info_real

