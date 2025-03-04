import sys
import os
import numpy as np
import sklearn.linear_model as skl
import pickle
import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7]

train_path = 'cache/processed_data/BIGMEG1/train_thingsmeg_sub-BIGMEG1.npy'
train_meg = np.load(train_path, mmap_mode='r')
# train_meg = train_meg[:8000,:,:]
train_meg = train_meg.reshape(train_meg.shape[0], -1)
test_path = 'cache/processed_data/BIGMEG1/test_thingsmeg_sub-BIGMEG1.npy'
test_meg = np.load(test_path, mmap_mode='r')
# test_meg = test_meg[:1000,:,:]
test_meg = test_meg.reshape(test_meg.shape[0], -1)
print(train_meg.shape, test_meg.shape)

## Preprocessing fMRI

train_fmri = train_meg
test_fmri = test_meg


norm_mean_train = np.mean(train_fmri, axis=0)
norm_scale_train = np.std(train_fmri, axis=0, ddof=1)
train_fmri = (train_fmri - norm_mean_train) / norm_scale_train
test_fmri = (test_fmri - norm_mean_train) / norm_scale_train

print(np.mean(train_fmri),np.std(train_fmri))
print(np.mean(test_fmri),np.std(test_fmri))

print(np.max(train_fmri),np.min(train_fmri))
print(np.max(test_fmri),np.min(test_fmri))

num_voxels, num_train, num_test = train_fmri.shape[1], len(train_fmri), len(test_fmri)


train_clip = np.load('cache/extracted_embeddings/BIGMEG1/train_cliptext_sub-BIGMEG1.npy', mmap_mode='r')
test_clip = np.load('cache/extracted_embeddings/BIGMEG1/test_cliptext_sub-BIGMEG1.npy', mmap_mode='r')
# train_clip = train_clip[:8000,:,:]
# test_clip = test_clip[:1000,:,:]

## Regression
num_samples,num_embed,num_dim = train_clip.shape

print("Training Regression")
reg_w = np.zeros((num_embed,num_dim,num_voxels)).astype(np.float32)
reg_b = np.zeros((num_embed,num_dim)).astype(np.float32)
pred_clip = np.zeros_like(test_clip)
for i in range(num_embed):
    reg = skl.Ridge(alpha=100000, max_iter=50000, fit_intercept=True)
    reg.fit(train_fmri, train_clip[:,i])
    reg_w[i] = reg.coef_
    reg_b[i] = reg.intercept_
    
    pred_test_latent = reg.predict(test_fmri)
    std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latent,axis=0)) / np.std(pred_test_latent,axis=0)
    pred_clip[:,i] = std_norm_test_latent * np.std(train_clip[:,i],axis=0) + np.mean(train_clip[:,i],axis=0)
    print(i,reg.score(test_fmri,test_clip[:,i]))
# 15 0.14176179041169176
# 16 0.14576387719203548
# 17 0.14871457869596197
# 18 0.1505361327954745
# 19 0.15149696882159835
# 20 0.15169051619959698
# 21 0.15187399518307262
# 22 0.151809517440021
# 23 0.15190029556854132


# np.save('data/predicted_features/subj{:02d}/nsd_cliptext_predtest_nsdgeneral.npy'.format(sub),pred_clip)
# np.save('data/predicted_features/subj{:02d}/nsd_cliptext_predtest_nsdgeneral_assumehrf.npy'.format(sub),pred_clip)
subject = 'BIGMEG1'
save_dir = 'cache/predicted_embeddings/' + subject + '/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
np.save(save_dir + f'thingsmeg_regress_cliptext_sub-{subject}.npy', pred_clip)

datadict = {
    'weight' : reg_w,
    'bias' : reg_b,

}

# with open('data/regression_weights/subj{:02d}/cliptext_regression_weights.pkl'.format(sub),"wb") as f:
#   pickle.dump(datadict,f)
# with open('data/regression_weights/subj{:02d}/cliptext_regression_weights_assumehrf.pkl'.format(sub),"wb") as f:
#   pickle.dump(datadict,f)
subject = 'BIGMEG1'
save_dir = 'cache/regression_weights/' + subject + '/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
with open(save_dir + f'thingsmeg_regress_cliptext_weights_sub-{subject}.pkl', "wb") as f:
    pickle.dump(datadict,f)