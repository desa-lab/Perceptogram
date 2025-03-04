import cortex
from nsd_mapdata import NSDmapdata

import os
os.environ['FREESURFER_HOME'] = '/usr/local/freesurfer/7.4.1'
os.environ['PATH'] = os.pathsep.join([os.path.join('/usr/local/freesurfer/7.4.1', 'bin'), os.environ['PATH']])
os.environ['SUBJECTS_DIR'] = 'data/nsd_preproc/freesurfer'

print(cortex.database.default_filestore)

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
parser.add_argument('-pattern', '--pattern-type', help='Pattern Type', default='clip')
args = parser.parse_args()
sub = int(args.sub)
pattern_type = args.pattern_type
assert pattern_type in ['clip', 'pca-brightness', 'ica-color', 'vdvae-texture']

base_path = os.path.join('data/nsd_preproc')
nsd = NSDmapdata(base_path)
sourcespace = 'func1pt8'
sourcefolder = f'cache/nsd_preproc/predicted_patterns/{pattern_type}_patterns/sub-{sub:02d}/func1pt8mm/'
outputfolder = f'cache/nsd_preproc/predicted_patterns/{pattern_type}_patterns/sub-{sub:02d}/mni/'

os.makedirs(outputfolder, exist_ok=True)
for filename in os.listdir(sourcefolder):
    sourcedata = os.path.join(sourcefolder, filename)
    targetspace = 'MNI'
    targetdata = nsd.fit(
        sub,
        sourcespace,
        targetspace,
        sourcedata,
        interptype='cubic',
        badval=0,
        outputfile=os.path.join(outputfolder, filename))