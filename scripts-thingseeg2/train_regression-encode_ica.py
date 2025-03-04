import numpy as np
from scipy.spatial.distance import correlation
import sklearn.linear_model as skl
import os
import pickle

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
parser.add_argument('-weights', '--saving_weights',help="Saving the weights", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument('-size', '--size', help='Size', default=16540)
parser.add_argument('-avg', '--average', help='Number of averages', default='')
parser.add_argument('-duration', '--duration', help='Duration', default=80)
parser.add_argument('-mirrored', '--mirrored', help='Mirrored electrode locations along the midline for the test set', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('-mirrored2', '--mirrored2', help='Mirrored electrode locations along the midline for the test set only for O1 and O2', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('-half', '--half', help='Half of the channels using the 10-20 instead of 10-10 montage', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('-dsi1', '--dsi1', help='Simulate DSI-24 layout, using P7 and P8', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('-dsi2', '--dsi2', help='Simulate DSI-24 layout, not using P7 and P8', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('-alpha', '--alpha', help='Alpha for regression strength', default=1000)
parser.add_argument('-param', '--param', help='Custom Parameter', default='')

args = parser.parse_args()
sub=int(args.sub)
saving_weights=args.saving_weights
alpha = int(args.alpha)
train_size=int(args.size)
average=args.average
duration=int(args.duration)
mirrored=args.mirrored
mirrored2=args.mirrored2
half=args.half
dsi1=args.dsi1
dsi2=args.dsi2
if average != '' or train_size != 16540 or duration != 80:
    param = f'_{train_size}avg{average}_dur{duration}'
else:
    param = ''
custom_param=args.param
if custom_param != '':
    param += f'_{custom_param}'

# Load EEG data
if duration != 0:
    eeg_train = np.load(f'data/thingseeg2_preproc/sub-{sub:02d}/train_thingseeg2_avg.npy')[:train_size,:,:duration]
    eeg_test = np.load(f'data/thingseeg2_preproc/sub-{sub:02d}/test_thingseeg2_avg{average}.npy')[:,:,:duration]
else:
    eeg_train = np.load(f'data/thingseeg2_preproc/sub-{sub:02d}/train_thingseeg2_avg_null.npy')[:train_size]
    eeg_test = np.load(f'data/thingseeg2_preproc/sub-{sub:02d}/test_thingseeg2_avg{average}_null.npy')
if mirrored:
    mirrored_channel_inds = [0, 6, 7, 5, 4, 3, 1, 2, 16, 15, 14, 13, 12, 11, 10, 9, 8]
    eeg_test = eeg_test[:, mirrored_channel_inds, :]
    param += '_mirrored'
elif mirrored2:
    mirrored_channel_inds = [0, 1, 2, 5, 4, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] # mirrored O1 and O2
    eeg_test = eeg_test[:, mirrored_channel_inds, :]
    param += '_mirrored2'
elif half:
    eeg_train = eeg_train[:, :8, :]
    eeg_test = eeg_test[:, :8, :]
    param += '_half'
elif dsi1:
    dsi1_channel_inds = [0, 1, 2, 3, 5, 6, 7]
    eeg_train = eeg_train[:, dsi1_channel_inds, :]
    eeg_test = eeg_test[:, dsi1_channel_inds, :]
    param += '_dsi1'
elif dsi2:
    dsi2_channel_inds = [0,1,3,4,5]
    eeg_train = eeg_train[:, dsi2_channel_inds, :]
    eeg_test = eeg_test[:, dsi2_channel_inds, :]
    param += '_dsi2'
eeg_train = eeg_train.reshape(eeg_train.shape[0],-1)
eeg_test = eeg_test.reshape(eeg_test.shape[0],-1)
print(eeg_train.shape, eeg_test.shape)
norm_mean_train = np.mean(eeg_train, axis=0)
norm_scale_train = np.std(eeg_train, axis=0, ddof=1)
eeg_train = (eeg_train - norm_mean_train) / norm_scale_train
eeg_test = (eeg_test - norm_mean_train) / norm_scale_train


# Save Directory
weights_save_dir = f'cache/thingseeg2_preproc/regression_weights/sub-{sub:02d}/'
if not os.path.exists(weights_save_dir):
    os.makedirs(weights_save_dir)
weights_filename = f'regress-encode_ica1k_weights{param}.pkl'
save_dir = f'cache/thingseeg2_preproc/predicted_embeddings/sub-{sub:02d}/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
pred_filename = f'regress-encode_ica1k{param}.npy'

ids = list(range(len(eeg_train)))
# Regression
train_latents= np.load('cache/thingseeg2_extracted_embeddings/train_ica1k.npy', mmap_mode='r')[ids]
test_latents = np.load('cache/thingseeg2_extracted_embeddings/test_ica1k.npy', mmap_mode='r')
print(train_latents.shape, test_latents.shape)

train_latents_mean = np.mean(train_latents,axis=0)
train_latents_std = np.std(train_latents,axis=0)
train_latents = (train_latents - train_latents_mean) / train_latents_std
test_latents = (test_latents - train_latents_mean) / train_latents_std

print("Training Regression")
reg = skl.Ridge(alpha=alpha, max_iter=50000, fit_intercept=True)
reg.fit(train_latents, eeg_train)
print('Training complete')

if saving_weights:
    datadict = {
        'weight' : reg.coef_,
        'bias' : reg.intercept_,
    }

    with open(weights_save_dir + weights_filename, "wb") as f:
        pickle.dump(datadict,f)

pred_eeg = reg.predict(test_latents)

np.save(save_dir + pred_filename, pred_eeg)

# Compute the Euclidean distances
euclidean_distances = np.array([np.linalg.norm(u - v) for u, v in zip(pred_eeg, eeg_test)])
correlation_distances = np.array([correlation(u, v) for u, v in zip(pred_eeg, eeg_test)])
# Compute the average Euclidean distance
average_euclidean_distance = euclidean_distances.mean()
correlations = (1 - correlation_distances).mean()
print(reg.score(test_latents, eeg_test), average_euclidean_distance, correlations)

# -1.1317343127577941 13.595983878502054 0.05468043790343973 for alpha=1000
# -0.18452059931953008 10.945475819825806 0.058097234242863734 for alpha=1000000
# -0.1851138319175139 10.95175589859752 0.05819316762828041 when alpha=10000000