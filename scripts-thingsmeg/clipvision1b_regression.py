import sys
import os
import numpy as np
import sklearn.linear_model as skl
import pickle
import argparse
from scipy.spatial.distance import correlation
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7]

# train_path = 'cache/thingsmeg/processed_data/BIGMEG1/train_thingsmeg_sub-BIGMEG1.npy'
# train_path = 'cache/thingsmeg/processed_data/BIGMEG1/train_thingsmeg_sub-BIGMEG1_200ms.npy'
# train_path = 'cache/thingsmeg/processed_data/BIGMEG1/train_thingsmeg_sub-BIGMEG1_400ms.npy'
# train_path = 'cache/thingsmeg/processed_data/BIGMEG1/train_thingsmeg_sub-BIGMEG1_600ms.npy'
train_path = 'cache/thingsmeg/processed_data/BIGMEG1/train_thingsmeg_sub-BIGMEG1_800ms.npy'
train_meg = np.load(train_path, mmap_mode='r')
# train_meg = train_meg[:8000,:,:]
train_meg = train_meg.reshape(train_meg.shape[0], -1)
# test_path = 'cache/thingsmeg/processed_data/BIGMEG1/test_thingsmeg_sub-BIGMEG1.npy'
# test_path = 'cache/thingsmeg/processed_data/BIGMEG1/test_thingsmeg_sub-BIGMEG1_200ms.npy'
# test_path = 'cache/thingsmeg/processed_data/BIGMEG1/test_thingsmeg_sub-BIGMEG1_400ms.npy'
# test_path = 'cache/thingsmeg/processed_data/BIGMEG1/test_thingsmeg_sub-BIGMEG1_600ms.npy'
test_path = 'cache/thingsmeg/processed_data/BIGMEG1/test_thingsmeg_sub-BIGMEG1_800ms.npy'
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


train_clip = np.load('cache/thingsmeg/extracted_embeddings/BIGMEG1/train_clipvision1b_sub-BIGMEG1.npy', mmap_mode='r')
test_clip = np.load('cache/thingsmeg/extracted_embeddings/BIGMEG1/test_clipvision1b_sub-BIGMEG1.npy', mmap_mode='r')
# train_clip = train_clip[:8000,:,:]
# test_clip = test_clip[:1000,:,:]

#train_clip = train_clip[:,1:,:]
num_samples,num_embed,num_dim = train_clip.shape

print("Training Regression")
reg_w = np.zeros((num_embed,num_dim,num_voxels)).astype(np.float32)
reg_b = np.zeros((num_embed,num_dim)).astype(np.float32)
pred_clip = np.zeros_like(test_clip)
for i in range(num_embed):


    reg = skl.Ridge(alpha=60000, max_iter=50000, fit_intercept=True) # old alpha=60000, optimal alpha=300000
    reg.fit(train_fmri, train_clip[:,i])
    reg_w[i] = reg.coef_
    reg_b[i] = reg.intercept_
    
    pred_test_latent = reg.predict(test_fmri)
    std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latent,axis=0)) / np.std(pred_test_latent,axis=0)
    pred_clip[:,i] = std_norm_test_latent * np.std(train_clip[:,i],axis=0) + np.mean(train_clip[:,i],axis=0)

    # Compute the Euclidean distances
    euclidean_distances = np.array([np.linalg.norm(u - v) for u, v in zip(pred_clip[:,i], test_clip[:,i])])
    correlation_distances = np.array([correlation(u, v) for u, v in zip(pred_clip[:,i], test_clip[:,i])])
    # Compute the average Euclidean distance
    average_euclidean_distance = euclidean_distances.mean()
    correlations = (1 - correlation_distances).mean()
    
    print(i,reg.score(test_fmri,test_clip[:,i]), average_euclidean_distance, correlations)
    

# np.save('data/predicted_features/subj{:02d}/nsd_clipvision_predtest_nsdgeneral.npy'.format(sub),pred_clip)
    # np.save('data/predicted_features/subj{:02d}/nsd_clipvision_predtest_nsdgeneral_assumehrf.npy'.format(sub),pred_clip)
subject = 'BIGMEG1'
save_dir = 'cache/thingsmeg/predicted_embeddings/' + subject + '/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# np.save(save_dir + f'thingsmeg_regress_clipvision1b_sub-{subject}.npy', pred_clip)
# np.save(save_dir + f'thingsmeg_regress_clipvision1b_sub-{subject}_200ms.npy', pred_clip)
# np.save(save_dir + f'thingsmeg_regress_clipvision1b_sub-{subject}_400ms.npy', pred_clip)
# np.save(save_dir + f'thingsmeg_regress_clipvision1b_sub-{subject}_600ms.npy', pred_clip)
np.save(save_dir + f'thingsmeg_regress_clipvision1b_sub-{subject}_800ms.npy', pred_clip)


datadict = {
    'weight' : reg_w,
    'bias' : reg_b,

}

# with open('data/regression_weights/subj{:02d}/clipvision_regression_weights.pkl'.format(sub),"wb") as f:
# with open('data/regression_weights/subj{:02d}/clipvision_regression_weights_assumehrf.pkl'.format(sub),"wb") as f:
#   pickle.dump(datadict,f)
subject = 'BIGMEG1'
save_dir = 'cache/thingsmeg/regression_weights/' + subject + '/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# with open(save_dir + f'thingsmeg_regress_clipvision1b_weights_sub-{subject}.pkl', "wb") as f:
#     pickle.dump(datadict,f)
# with open(save_dir + f'thingsmeg_regress_clipvision1b_weights_sub-{subject}_200ms.pkl', "wb") as f:
#     pickle.dump(datadict,f)
# with open(save_dir + f'thingsmeg_regress_clipvision1b_weights_sub-{subject}_400ms.pkl', "wb") as f:
#     pickle.dump(datadict,f)
# with open(save_dir + f'thingsmeg_regress_clipvision1b_weights_sub-{subject}_600ms.pkl', "wb") as f:
#     pickle.dump(datadict,f)
with open(save_dir + f'thingsmeg_regress_clipvision1b_weights_sub-{subject}_800ms.pkl', "wb") as f:
    pickle.dump(datadict,f)
    
# 0 9.471119049517251e-07 0.878658145980226 0.6127160358947972                                                                           
# 1 -0.0213864995796117 0.6405271175013325 0.4857552759397123                                                                            
# 2 -0.021646810261362604 0.6668038936966573 0.4788284850817272                                                                          
# 3 -0.022493003664171233 0.6644990830307966 0.47081091664398533                                                                         
# 4 -0.02170905249980429 0.664453547451643 0.47207231906625147                                                                           
# 5 -0.021428790272931875 0.6654098310243959 0.47027966643213615                                                                         
# 6 -0.020698950861931326 0.66837281673314 0.4709815522468214                                                                            
# 7 -0.023762048575726547 0.6816671236957403 0.460289372962115                                                                           
# 8 -0.0213884457787473 0.6728269088879799 0.4632624701452484                                                                            
# 9 -0.022915423660775303 0.6752677950392509 0.4597034056616972                                                                          
# 10 -0.023535264374676193 0.6775911090003987 0.4545390972197049                                                                         
# 11 -0.020097215745443587 0.6756944370644618 0.46143716455870654                                                                        
# 12 -0.021462701891026476 0.6761936861141966 0.4645273055954277                                                                         
# 13 -0.020795021531669868 0.6697752868815365 0.4690782382098659