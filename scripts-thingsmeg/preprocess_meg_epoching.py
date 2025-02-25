import mne
from mne import concatenate_epochs, concatenate_raws
# mne.viz.set_3d_backend("notebook")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids
from itertools import product
from tqdm import tqdm
import os
import os.path as osp

data_dir = '/mnt/sphere/projects/image_decoding_from_brain/THINGS-MEG-raw/download'
subject_file = data_dir + '/participants.tsv'
subjects = pd.read_csv(subject_file, sep="\t")
def get_subject_id(x):
    return x.split("-")[1]  # noqa
subjects = subjects.participant_id.apply(get_subject_id).values
subject = subjects[0]
tasks = ['main']
sessions = ['{:02d}'.format(x) for x in range(1,13)]  # 2 recording sessions
runs = ['{:02d}'.format(x) for x in range(1,11)]

def clipping(data, n_std = 5):
    a_min = data.mean() - n_std * data.std()
    a_max = data.mean() + n_std * data.std()
    return np.clip(data, a_min, a_max)

mag_ch_names = ['MLC11-1609','MLC12-1609','MLC13-1609','MLC14-1609','MLC15-1609','MLC16-1609','MLC17-1609','MLC21-1609','MLC22-1609','MLC23-1609','MLC24-1609','MLC25-1609','MLC31-1609','MLC32-1609','MLC41-1609','MLC42-1609','MLC51-1609','MLC52-1609','MLC53-1609','MLC54-1609','MLC55-1609','MLC61-1609','MLC62-1609','MLC63-1609',
 'MLF11-1609','MLF12-1609','MLF13-1609','MLF14-1609','MLF21-1609','MLF22-1609','MLF23-1609','MLF24-1609','MLF31-1609','MLF32-1609','MLF33-1609','MLF34-1609','MLF35-1609','MLF41-1609','MLF42-1609','MLF43-1609','MLF44-1609','MLF45-1609','MLF46-1609','MLF51-1609','MLF52-1609','MLF53-1609','MLF54-1609','MLF55-1609','MLF56-1609','MLF61-1609','MLF62-1609','MLF63-1609','MLF64-1609','MLF65-1609','MLF66-1609','MLF67-1609',
 'MLO11-1609','MLO12-1609','MLO13-1609','MLO14-1609','MLO21-1609','MLO22-1609','MLO23-1609','MLO24-1609','MLO31-1609','MLO32-1609','MLO33-1609','MLO34-1609','MLO41-1609','MLO42-1609','MLO43-1609','MLO44-1609','MLO51-1609','MLO52-1609','MLO53-1609',
 'MLP11-1609','MLP12-1609','MLP21-1609','MLP22-1609','MLP23-1609','MLP31-1609','MLP32-1609','MLP33-1609','MLP34-1609','MLP35-1609','MLP41-1609','MLP42-1609','MLP43-1609','MLP44-1609','MLP45-1609','MLP51-1609','MLP52-1609','MLP53-1609','MLP54-1609','MLP55-1609','MLP56-1609','MLP57-1609',
 'MLT11-1609','MLT12-1609','MLT13-1609','MLT14-1609','MLT15-1609','MLT16-1609','MLT21-1609','MLT22-1609','MLT23-1609','MLT24-1609','MLT25-1609','MLT26-1609','MLT27-1609','MLT31-1609','MLT32-1609','MLT33-1609','MLT34-1609','MLT35-1609','MLT36-1609','MLT37-1609','MLT41-1609','MLT42-1609','MLT43-1609','MLT44-1609','MLT45-1609','MLT46-1609','MLT47-1609','MLT51-1609','MLT52-1609','MLT53-1609','MLT54-1609','MLT55-1609','MLT56-1609','MLT57-1609',
 'MRC11-1609','MRC12-1609','MRC13-1609','MRC14-1609','MRC15-1609','MRC16-1609','MRC17-1609','MRC21-1609','MRC22-1609','MRC23-1609','MRC24-1609','MRC25-1609','MRC31-1609','MRC32-1609','MRC41-1609','MRC42-1609','MRC51-1609','MRC52-1609','MRC53-1609','MRC54-1609','MRC55-1609','MRC61-1609','MRC62-1609','MRC63-1609',
 'MRF11-1609','MRF12-1609','MRF13-1609','MRF14-1609','MRF21-1609','MRF22-1609','MRF23-1609','MRF24-1609','MRF25-1609','MRF31-1609','MRF32-1609','MRF33-1609','MRF34-1609','MRF35-1609','MRF41-1609','MRF42-1609','MRF44-1609','MRF45-1609','MRF46-1609','MRF51-1609','MRF52-1609','MRF53-1609','MRF54-1609','MRF55-1609','MRF56-1609','MRF61-1609','MRF62-1609','MRF63-1609','MRF64-1609','MRF65-1609','MRF66-1609','MRF67-1609',
 'MRO11-1609','MRO12-1609','MRO14-1609','MRO21-1609','MRO22-1609','MRO23-1609','MRO24-1609','MRO31-1609','MRO32-1609','MRO33-1609','MRO34-1609','MRO41-1609','MRO42-1609','MRO43-1609','MRO44-1609','MRO51-1609','MRO52-1609','MRO53-1609',
 'MRP11-1609', 'MRP12-1609', 'MRP21-1609', 'MRP22-1609', 'MRP23-1609', 'MRP31-1609', 'MRP32-1609', 'MRP33-1609', 'MRP34-1609', 'MRP35-1609', 'MRP41-1609', 'MRP42-1609', 'MRP43-1609', 'MRP44-1609', 'MRP45-1609', 'MRP51-1609', 'MRP52-1609', 'MRP53-1609', 'MRP54-1609', 'MRP55-1609', 'MRP56-1609','MRP57-1609',
 'MRT11-1609', 'MRT12-1609', 'MRT13-1609', 'MRT14-1609', 'MRT15-1609', 'MRT16-1609', 'MRT21-1609', 'MRT22-1609', 'MRT23-1609', 'MRT24-1609', 'MRT25-1609', 'MRT26-1609', 'MRT27-1609', 'MRT31-1609', 'MRT32-1609', 'MRT33-1609', 'MRT34-1609', 'MRT35-1609', 'MRT36-1609', 'MRT37-1609', 'MRT41-1609', 'MRT42-1609', 'MRT43-1609', 'MRT44-1609', 'MRT45-1609', 'MRT46-1609', 'MRT47-1609', 'MRT51-1609', 'MRT52-1609', 'MRT53-1609', 'MRT54-1609', 'MRT55-1609', 'MRT56-1609', 'MRT57-1609',
 'MZC01-1609', 'MZC02-1609', 'MZC03-1609', 'MZC04-1609',
 'MZF01-1609','MZF02-1609','MZF03-1609',
 'MZO01-1609','MZO02-1609','MZO03-1609',
 'MZP01-1609']
ch_type_map = {'UPPT001': 'stim'}
ch_type_map.update({x: 'mag' for x in mag_ch_names})

labels = np.load('data/things_text_labels.npy', mmap_mode='r')

if not osp.exists('cache/processed_data/{}/epochs/'.format(subject)):
    os.makedirs('cache/processed_data/{}/epochs/'.format(subject))

raw_list = []
for session, task, run in tqdm(product(sessions, tasks, runs), total=len(sessions)*len(tasks)*len(runs)):
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        root=data_dir,
        run = run,
        datatype="meg",
    )
    raw = read_raw_bids(bids_path, verbose='ERROR')
    raw.set_channel_types(ch_type_map, verbose='ERROR')
    raw_list.append(raw)
    
raws = concatenate_raws(raw_list, on_mismatch="ignore")
index1b_list = [int(float(i.split('/')[-1])) - 1 if '/' in i else -1 for i in raws.annotations.description]
combined_annot = np.array([item if item in ['catch', 'BAD boundary', 'EDGE boundary'] else f'{item.split("/")[0]}/{labels[index1b_list][i]}/{item.split("/")[1]}' for i, item in enumerate(raws.annotations.description)])
raws.annotations.rename(dict(zip(raws.annotations.description, combined_annot)))
_, all_event_dict = mne.events_from_annotations(raws, verbose='ERROR')

i_iter = 0
epoch_list = []
for session, task, run in tqdm(product(sessions, tasks, runs), total=len(sessions)*len(tasks)*len(runs)):
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        root=data_dir,
        run = run,
        datatype="meg",
    )
    raw = read_raw_bids(bids_path, verbose='ERROR')
    raw.set_channel_types(ch_type_map, verbose='ERROR')
    index1b_list = [int(float(i.split('/')[-1])) - 1 if '/' in i else -1 for i in raw.annotations.description]
    # combined_annot = np.array([item if item == 'catch' else f'{item.split("/")[0]}/{labels[index1b_list][i]}/{item.split("/")[1]}' for i, item in enumerate(raw.annotations.description)])
    combined_annot = np.array([item if item in ['catch', 'BAD boundary', 'EDGE boundary'] else f'{item.split("/")[0]}/{labels[index1b_list][i]}/{item.split("/")[1]}' for i, item in enumerate(raw.annotations.description)])
    raw.annotations.rename(dict(zip(raw.annotations.description, combined_annot)))
    # events, event_dict = mne.events_from_annotations(raw, verbose='ERROR')
    events, event_dict = mne.events_from_annotations(raw, verbose='ERROR', event_id=all_event_dict)
    epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.3, tmax=1.0, baseline=(-0.3, 0), preload=True, picks='mag', verbose='ERROR').resample(120, npad="auto", verbose='ERROR')
    # epochs.resample(120, npad="auto", verbose='ERROR', n_jobs='cuda')
    # epochs.resample(120, npad="auto", verbose='ERROR', n_jobs=100)
    # epochs.crop(tmin=0.0, tmax=epochs.tmax, verbose=False);
    # epochs.filter(0, 40, verbose=False, n_jobs='cuda');
    # epochs.filter(0, 40, verbose=False, n_jobs=100);
    epochs.filter(0, 40, verbose=False);
    epochs.apply_function(clipping, n_std = 5, verbose=False);
    epoch_list.append(epochs)
    # if i_iter == 5:
    #     break
    # i_iter += 1

all_epochs = concatenate_epochs(epoch_list, on_mismatch="ignore") # head positions are different in each run
all_epochs.crop(tmin=0.0, tmax=all_epochs.tmax, verbose=False);
print('cropped')

import pdb; pdb.set_trace()

all_epochs.save('cache/processed_data/BIGMEG1/epochs/epochs_meg_epo.fif.gz',split_naming='bids')