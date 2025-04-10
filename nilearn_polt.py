
import os
import numpy as np
import pandas as pd
import nibabel as nib
import go_path as paths
import n_tools as nt
import nilearn.plotting as nip
import nilearn.surface as nis
import h5pyio

h5py_file = os.path.join(paths.states_results_dir, 'lesion_surf_2_symsurf.h5py')
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