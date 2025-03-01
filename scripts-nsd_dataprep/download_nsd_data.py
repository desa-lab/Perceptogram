import os

# Download Experiment Infos
os.system('aws s3 cp s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_expdesign.mat data/nsd_metadata/experiments/nsd/ --no-sign-request')
os.system('aws s3 cp s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_stim_info_merged.pkl data/nsd_metadata/experiments/nsd/ --no-sign-request')

# Download Stimuli
os.system('aws s3 cp s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5 data/nsd_metadata/stimuli/nsd/ --no-sign-request')

# Download Betas
for sub in [1,2,5,7]:
    for sess in range(1,38):
        os.system('aws s3 cp s3://natural-scenes-dataset/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session{:02d}.nii.gz data/nsd_preproc/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/ --no-sign-request'.format(sub,sess,sub))

# Download ROIs
for sub in [1,2,5,7]:
    os.system('aws s3 cp s3://natural-scenes-dataset/nsddata/ppdata/subj{:02d}/func1pt8mm/roi/ data/nsd_preproc/subj{:02d}/func1pt8mm/roi/ --no-sign-request --recursive'.format(sub,sub))

# Download Freesurfer
os.system('aws s3 cp s3://natural-scenes-dataset/nsddata/freesurfer/ data/nsd_preproc/freesurfer/ --no-sign-request --recursive')
# for sub in [1,2,5,7]:
#     os.system('aws s3 cp s3://natural-scenes-dataset/nsddata/freesurfer/subj{:02d}/ data/nsd_preproc/subj{:02d}/freesurfer/ --no-sign-request --recursive'.format(sub,sub))

# Download MNI Transforms
for sub in [1,2,5,7]:
    os.system(f'aws s3 cp s3://natural-scenes-dataset/nsddata/ppdata/subj{sub:02d}/transforms/func1pt8-to-MNI.nii.gz data/nsd_preproc/nsddata/ppdata/subj{sub:02d}/transforms/func1pt8-to-MNI.nii.gz --no-sign-request')
