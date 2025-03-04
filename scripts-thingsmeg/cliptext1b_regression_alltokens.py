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


train_clip = np.load('cache/thingsmeg/extracted_embeddings/BIGMEG1/train_cliptext1bcategory_sub-BIGMEG1.npy', mmap_mode='r')
test_clip = np.load('cache/thingsmeg/extracted_embeddings/BIGMEG1/test_cliptext1bcategory_sub-BIGMEG1.npy', mmap_mode='r')
# train_clip = train_clip[:8000,:,:]
# test_clip = test_clip[:1000,:,:]

## Regression
num_samples,num_embed,num_dim = train_clip.shape

print("Training Regression")
reg_w = np.zeros((num_embed,num_dim,num_voxels)).astype(np.float32)
reg_b = np.zeros((num_embed,num_dim)).astype(np.float32)
pred_clip = np.zeros_like(test_clip)
for i in range(num_embed):
    reg = skl.Ridge(alpha=100000, max_iter=50000, fit_intercept=True) # old alpha=100000, optimal alpha=1200000
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



# np.save('data/predicted_features/subj{:02d}/nsd_cliptext_predtest_nsdgeneral.npy'.format(sub),pred_clip)
# np.save('data/predicted_features/subj{:02d}/nsd_cliptext_predtest_nsdgeneral_assumehrf.npy'.format(sub),pred_clip)
subject = 'BIGMEG1'
save_dir = 'cache/thingsmeg/predicted_embeddings/' + subject + '/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# np.save(save_dir + f'thingsmeg_regress_cliptext1bcategoryalltokens_sub-{subject}.npy', pred_clip)
# np.save(save_dir + f'thingsmeg_regress_cliptext1bcategoryalltokens_sub-{subject}_200ms.npy', pred_clip)
# np.save(save_dir + f'thingsmeg_regress_cliptext1bcategoryalltokens_sub-{subject}_400ms.npy', pred_clip)
# np.save(save_dir + f'thingsmeg_regress_cliptext1bcategoryalltokens_sub-{subject}_600ms.npy', pred_clip)
np.save(save_dir + f'thingsmeg_regress_cliptext1bcategoryalltokens_sub-{subject}_800ms.npy', pred_clip)

datadict = {
    'weight' : reg_w,
    'bias' : reg_b,

}

# with open('data/regression_weights/subj{:02d}/cliptext_regression_weights.pkl'.format(sub),"wb") as f:
#   pickle.dump(datadict,f)
# with open('data/regression_weights/subj{:02d}/cliptext_regression_weights_assumehrf.pkl'.format(sub),"wb") as f:
#   pickle.dump(datadict,f)
subject = 'BIGMEG1'
save_dir = 'cache/thingsmeg/regression_weights/' + subject + '/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# with open(save_dir + f'thingsmeg_regress_cliptext1bcategoryalltokens_weights_sub-{subject}.pkl', "wb") as f:
#     pickle.dump(datadict,f)
# with open(save_dir + f'thingsmeg_regress_cliptext1bcategoryalltokens_weights_sub-{subject}_200ms.pkl', "wb") as f:
#     pickle.dump(datadict,f)
# with open(save_dir + f'thingsmeg_regress_cliptext1bcategoryalltokens_weights_sub-{subject}_400ms.pkl', "wb") as f:
#     pickle.dump(datadict,f)
# with open(save_dir + f'thingsmeg_regress_cliptext1bcategoryalltokens_weights_sub-{subject}_600ms.pkl', "wb") as f:
#     pickle.dump(datadict,f)
with open(save_dir + f'thingsmeg_regress_cliptext1bcategoryalltokens_weights_sub-{subject}_800ms.pkl', "wb") as f:
    pickle.dump(datadict,f)


# 0 -0.006960497720738112 0.06821725544159211 0.9999999999999997                                
# 1 -0.016928714038937365 0.8211599703864206 0.5212055044038982                                         
# 2 -0.014297685250419774 0.8548719078063546 0.5606922130908927                                         
# 3 -0.010684331038373527 0.8024277822581044 0.6677438967426019                                         
# 4 -0.009888076947434056 0.783350493419263 0.6904277603611981                                          
# 5 -0.009850957783138414 0.7780343223868317 0.6926813174570677                                         
# 6 -0.009836090170500562 0.7739127222384612 0.6926871887936589
# 7 -0.010108329836819105 0.7690525703564015 0.6930063401147549
# 8 -0.01095032592322429 0.7646957382369671 0.6929958959197537
# 9 -0.011524299688184672 0.7593612010158538 0.6936360366683478
# 10 -0.011642593790238583 0.7548992370271785 0.6945015926004782
# 11 -0.011666911014807747 0.74994356474616 0.6954090214976827
# 12 -0.011763978526046182 0.7449310624165144 0.696982730885556
# 13 -0.011783770524103126 0.7413305446830113 0.6978594610316672
# 14 -0.011805062118063753 0.7368788796310723 0.6992864737184603
# 15 -0.011754915216514533 0.7335550976088611 0.7005994294691656
# 16 -0.011641031984516258 0.7302130142990267 0.7014534403242538
# 17 -0.011393422584720465 0.7272356099993411 0.7021389472253192
# 18 -0.011179801607569956 0.7254090199081401 0.7023328025938901
# 19 -0.0109907457258639 0.7233875850478401 0.7026146079421123
# 20 -0.010889921322702882 0.7219550637402065 0.7033722236190025
# 21 -0.010753913623711829 0.7206070078228322 0.7038767583815714
# 22 -0.0106022511158679 0.7189344022945494 0.7046900807931226
# 23 -0.010464275199902284 0.7174803217321607 0.7053296037735884
# 24 -0.010338722487621502 0.7158284445646289 0.7054962497601455
# 25 -0.010247470648478274 0.7145271460763032 0.7055999096907506
# 26 -0.010157074438134834 0.7136896714029581 0.704975105445015
# 27 -0.010093743020228437 0.7126483729768028 0.70451316048876
# 28 -0.010012230246666642 0.7120238814290948 0.7040476369290509
# 29 -0.00996048169651104 0.711079625632479 0.7034498549604229
# 30 -0.00991102341089811 0.7103113522168139 0.7033149394089613
# 31 -0.009858833298212903 0.7099377309499392 0.7028290536071399
# 32 -0.009829389827966094 0.7092720424955393 0.7027884378374951
# 33 -0.009771771590963818 0.7086651813505798 0.7029320841382706
# 34 -0.009735052270356608 0.707491767142495 0.7029025084031585
# 35 -0.009693002014932227 0.706303585295668 0.7030928042335183
# 36 -0.009624538944071037 0.705755243198726 0.7023521586560132
# 37 -0.009589037606380545 0.7045594371713004 0.7018492453599305
# 38 -0.009533006704595177 0.7039081742219152 0.701654167761748
# 39 -0.00948479725030998 0.7031690774094146 0.7011832641580795
# 40 -0.009453351701961968 0.702596137491027 0.7011010446436713
# 41 -0.009421374079191606 0.7020315396478781 0.7015293506525825
# 42 -0.00941344536167662 0.7011593722911613 0.7016172226926429
# 43 -0.009399764838093572 0.699946295581465 0.7023709210862205
# 44 -0.009396400525706691 0.6985797072592561 0.7032724817092298
# 45 -0.009377708482653074 0.6975202618090192 0.7040988583502177
# 46 -0.009375055578185054 0.6960788231941111 0.7047936619239291
# 47 -0.009353587056484489 0.6949570754497563 0.7049841798054163
# 48 -0.00933381757510147 0.6938318529056834 0.704641864566636
# 49 -0.009307844814335997 0.6929234036680608 0.704800640494052
# 50 -0.009279774360884804 0.6922731415809225 0.7052028197223941
# 51 -0.009280080212266445 0.69137964839109 0.705671382433094
# 52 -0.009263724435377275 0.6908489027516709 0.7061196047106176
# 53 -0.009278899452095991 0.6895176798109531 0.7067299583256473
# 54 -0.00927444982352515 0.6883373678276198 0.707116856260123
# 55 -0.009278911445029544 0.6865868083908276 0.7082359519603246
# 56 -0.009289303557585917 0.6850575010255048 0.7091663003130342
# 57 -0.009276183130666355 0.683908071074589 0.7096672814121098
# 58 -0.009299258307837071 0.6823531432231325 0.7101320005251207
# 59 -0.0092802958433552 0.6813729657319006 0.7110064931102972
# 60 -0.009265201459342773 0.6798937206335247 0.7124208779192446
# 61 -0.009289428417554297 0.6784173091119773 0.7132928255477546
# 62 -0.009238872755372702 0.6772737455466823 0.7144045846303847
# 63 -0.00924667194885799 0.6754550130286653 0.7156833810344214
# 64 -0.009252333048933538 0.6742853454112683 0.7164267691301898
# 65 -0.009252845925065574 0.6725947947867205 0.7175890670225598
# 66 -0.009253105578456685 0.6712635544096671 0.7177749371272671
# 67 -0.009241533828537908 0.6697997324336383 0.7181853611429022
# 68 -0.009241684623659924 0.6684302056119911 0.7183307805574933
# 69 -0.009181384106630101 0.668113602754081 0.7183575420375382
# 70 -0.009148519548233957 0.6675256889173091 0.7185253239863097
# 71 -0.009183172255771524 0.666671757268751 0.7190908113878836
# 72 -0.00912703925650398 0.6655022414095946 0.7198248608040422
# 73 -0.009164918621283017 0.6642528195938969 0.7205994427773675
# 74 -0.009152108927428192 0.6632519057093335 0.7216008774598441
# 75 -0.009177750964329632 0.6619946568493584 0.7205637065144704
# 76 -0.00890455714478852 0.6604249547769592 0.7274809010879606