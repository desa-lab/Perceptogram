import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import cortex
import os

vol_dir = f'cache/nsd_preproc/predicted_patterns/ica-color_patterns/sub-01/mni/'
vol_filenames = os.listdir(vol_dir)
pattern_names = [vol_filename.split('_')[0] for vol_filename in vol_filenames]

output_folder = f'results/nsd_preproc/avg-1-2-5-7/ica-color_patterns/mni/'
os.makedirs(output_folder, exist_ok=True)
os.makedirs('cache/nsd_preproc/predicted_patterns/ica-color_patterns/avg-1-2-5-7/mni/', exist_ok=True)

for i, vol_filename in enumerate(vol_filenames):
    subs = [1,2,5,7]
    volumes = []
    for sub in subs:
        vol_dir = f'cache/nsd_preproc/predicted_patterns/ica-color_patterns/sub-{sub:02d}/mni/'
        vol_data = nib.load(vol_dir+vol_filename)
        volumes.append(vol_data.get_fdata())
    volumes = np.array(volumes)
    avg_vol = np.mean(volumes, axis=0)
    pattern_image = nib.Nifti1Image(avg_vol, affine=vol_data.affine, header=vol_data.header)
    nib.save(pattern_image, f'cache/nsd_preproc/predicted_patterns/ica-color_patterns/avg-1-2-5-7/mni/{vol_filename}')
    volumn = cortex.Volume(np.moveaxis(np.moveaxis(avg_vol,0,-1),0,1), subject='fsaverage', xfmname='atlas', cmap='twilight', vmin=-0.1, vmax=0.1)
    fig = plt.figure() # 100
    cortex.quickflat.make_figure(volumn,recache=1, fig=fig)
    plt.title(f'{pattern_names[i].capitalize()} Pattern - NSD 4-Subject Average (1, 2, 5, 7) MNI')
    plt.savefig(output_folder+vol_filename.replace('.nii.gz','.png'), dpi=300) # dpi=100 if uploading to github
    plt.close()

