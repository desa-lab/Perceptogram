import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import cortex
import os

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub = int(args.sub)

vol_dir = f'cache/nsd_preproc/predicted_patterns/vdvae-texture_patterns/sub-{sub:02d}/mni/'

output_folder = f'results/nsd_preproc/sub-{sub:02d}/vdvae-texture_patterns/mni/'
os.makedirs(output_folder, exist_ok=True)

cortex.download_subject('fsaverage')

vol_filenames = os.listdir(vol_dir)
for vol_filename in vol_filenames:
    vol_data = nib.load(vol_dir+vol_filename)
    volumn = cortex.Volume(np.moveaxis(np.moveaxis(vol_data.get_fdata(),0,-1),0,1), subject='fsaverage', xfmname='atlas', cmap='twilight', vmin=-0.4, vmax=0.4)
    fig = plt.figure() # 100
    cortex.quickflat.make_figure(volumn,recache=1, fig=fig)
    plt.savefig(output_folder+vol_filename.replace('.nii.gz','.png'), dpi=300)
    plt.close()


