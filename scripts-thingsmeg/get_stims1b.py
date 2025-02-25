# %%
import mne
# mne.viz.set_browser_backend("matplotlib")
# mne.viz.set_3d_backend("notebook")
from matplotlib import pyplot as plt
# %config InlineBackend.figure_format='retina'
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids
from itertools import product
from tqdm import tqdm
import os

# %%
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

# %%
print('Loading stims...')

# %%
images = np.load('data/things_stim_images.npy', mmap_mode='r')
text = np.load('data/things_text_labels.npy', mmap_mode='r')

# %%
print('stim shape: ', images.shape)

# %%
# train_images = None
# test_images = None
# train_text = None
# test_text = None
train_stim_ids = None
test_stim_ids = None
for session, task, run in tqdm(product(sessions, tasks, runs), total=len(sessions)*len(tasks)*len(runs), desc=f'Processing subject {subject}'):
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        root=data_dir,
        run = run,
        datatype="meg",
    )
    raw = read_raw_bids(bids_path, verbose='ERROR')
    events = list()
    for annot in raw.annotations:
        # event = eval(annot.pop("description"))
        event = dict()
        event['kind'] = annot['description'].split('/')[0]
        if annot['description'].split('/')[0] != 'catch':
            event['image_id'] = annot['description'].split('/')[1]
        else:
            event['image_id'] = '-1' # catch trials, maybe we should change this
        event['start'] = annot['onset']
        event['duration'] = annot['duration']
        events.append(event)
    events_df = pd.DataFrame(events)
    train_events_df = events_df[events_df['kind'] == 'exp']
    test_events_df = events_df[events_df['kind'] == 'test']
    if train_stim_ids is None:
        train_stim_ids = pd.to_numeric(train_events_df['image_id']).astype(int).to_numpy() -1 # -1 if we need one-based indexing
        test_stim_ids = pd.to_numeric(test_events_df['image_id']).astype(int).to_numpy() -1 # -1 if we need one-based indexing
    else:
        train_stim_ids = np.concatenate((train_stim_ids, pd.to_numeric(train_events_df['image_id']).astype(int).to_numpy() -1), axis=0)
        test_stim_ids = np.concatenate((test_stim_ids, pd.to_numeric(test_events_df['image_id']).astype(int).to_numpy() -1), axis=0)
    # train_image_ids = pd.to_numeric(train_events_df['image_id']).astype(int).to_numpy() # +1 if we need one-based indexing
    # test_image_ids = pd.to_numeric(test_events_df['image_id']).astype(int).to_numpy() # +1 if we need one-based indexing
    # if train_images is None:
    #     train_images = images[train_image_ids]
    #     test_images = images[test_image_ids]
    #     train_text = text[train_image_ids]
    #     test_text = text[test_image_ids]
    # else:
    #     train_images = np.concatenate((train_images, images[train_image_ids]), axis=0)
    #     test_images = np.concatenate((test_images, images[test_image_ids]), axis=0)
    #     train_text = np.concatenate((train_text, text[train_image_ids]), axis=0)
    #     test_text = np.concatenate((test_text, text[test_image_ids]), axis=0)
train_images = images[train_stim_ids]
test_images = images[test_stim_ids]
train_text = text[train_stim_ids]
test_text = text[test_stim_ids]

print('train_images shape: ', train_images.shape) # (22248, 257, 768)
print('test_images shape: ', test_images.shape) # (2400, 257, 768)

# %%
save_dir = 'cache/thingsmeg/processed_data/' + subject + '/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
np.save(save_dir + f'train_images1b_sub-{subject}.npy', train_images)
np.save(save_dir + f'test_images1b_sub-{subject}.npy', test_images)
np.save(save_dir + f'train_text1b_sub-{subject}.npy', train_text)
np.save(save_dir + f'test_text1b_sub-{subject}.npy', test_text)
