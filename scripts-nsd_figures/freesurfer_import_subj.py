import cortex
import os
import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub = int(args.sub)
os.environ['FREESURFER_HOME'] = '/usr/local/freesurfer/7.4.1'
os.environ['PATH'] = os.pathsep.join([os.path.join('/usr/local/freesurfer/7.4.1', 'bin'), os.environ['PATH']])
os.environ['SUBJECTS_DIR'] = 'data/nsd_preproc/freesurfer/'
freesurfer_path = 'data/nsd_preproc/freesurfer/'

subject = cortex.freesurfer.import_subj(f"subj{sub:02d}",freesurfer_subject_dir=freesurfer_path)
cortex.freesurfer.import_flat(f"subj{sub:02d}",'full',freesurfer_subject_dir=freesurfer_path,auto_overwrite=True)
pts_lh,polys_lh,_=cortex.freesurfer.get_surf(f'subj{sub:02d}','lh','patch','full'+'.flat',freesurfer_subject_dir=freesurfer_path) 
pts_rh,polys_rh,_=cortex.freesurfer.get_surf(f'subj{sub:02d}','rh','patch','full'+'.flat',freesurfer_subject_dir=freesurfer_path)
ref_path = f'data/nsd_preproc/subj{sub:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session01.nii.gz'
cortex.align.automatic(f'subj{sub:02d}','full',reference=ref_path)