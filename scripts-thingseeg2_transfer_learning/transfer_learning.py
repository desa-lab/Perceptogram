import numpy as np
import scipy
from scipy.spatial.distance import correlation
import random
import sklearn.linear_model as skl
import os
import pickle

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
parser.add_argument('-weights', '--saving_weights',help="Saving the weights", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("-source", "--transfer_source",help="Transfer source",default=2)
parser.add_argument("-size", "--size",help="Size",default=3000)
parser.add_argument("-method", "--method",help="Transfer Learning Method",default='augment') # augment or transfer
parser.add_argument('-bins', '--timebins',help="Mapping within Timebins",default=False, action=argparse.BooleanOptionalAction)
args = parser.parse_args()
sub=int(args.sub)
saving_weights=args.saving_weights
transfer_source=int(args.transfer_source) if args.transfer_source != 'other' else 'other'
common_size=int(args.size)
method=args.method
timebins=args.timebins

if method == 'augment':
    if transfer_source != 'other':
        eeg_common = np.load(f'data/thingseeg2_preproc/sub-{sub:02d}/train_thingseeg2_avg.npy')
        original_shape = eeg_common.shape
        eeg_common = eeg_common[:common_size]
        eeg_common = eeg_common.reshape(eeg_common.shape[0],-1)
        # if transfer_source == 'other':
        #     eeg_transfer_from = np.load(f'cache/thingseeg2_preproc/transfer/sub-{sub:02d}/train_thingseeg2_avg_other.npy')
        # else:
        eeg_transfer_from = np.load(f'data/thingseeg2_preproc/sub-{transfer_source:02d}/train_thingseeg2_avg.npy')
        eeg_common_transfer_from = eeg_transfer_from[:common_size]
        eeg_common_transfer_from = eeg_common_transfer_from.reshape(eeg_common_transfer_from.shape[0],-1)
        eeg_new_transfer_from = eeg_transfer_from[common_size:]
        eeg_new_transfer_from = eeg_new_transfer_from.reshape(eeg_new_transfer_from.shape[0],-1)

        print(eeg_common.shape, eeg_new_transfer_from.shape)

        print("Training Augmentation Regression")
        reg = skl.Ridge(alpha=1000, max_iter=50000, fit_intercept=True)
        reg.fit(eeg_common_transfer_from, eeg_common)
        print('Augmentation training complete')
        weights_save_dir = f'cache/thingseeg2_preproc/transfer_regression_weights/sub-{sub:02d}/'
        weights_filename = f'regress_weights_{common_size}avg_transfered_from{transfer_source}.pkl'
        if not os.path.exists(weights_save_dir):
            os.makedirs(weights_save_dir)
        if saving_weights:
            datadict = {
                'weight' : reg.coef_,
                'bias' : reg.intercept_,
            }
            with open(weights_save_dir + weights_filename, "wb") as f:
                pickle.dump(datadict,f)

        eeg_new_pred = reg.predict(eeg_new_transfer_from)
        eeg_transfered = np.concatenate((eeg_common, eeg_new_pred), axis=0)
        eeg_transfered = eeg_transfered.reshape(original_shape)
        eeg_nontransfered = np.concatenate((eeg_common, eeg_new_transfer_from), axis=0)
        eeg_nontransfered = eeg_nontransfered.reshape(original_shape)

        if not os.path.exists(f'cache/thingseeg2_preproc/transfer/sub-{sub:02d}'):
            os.makedirs(f'cache/thingseeg2_preproc/transfer/sub-{sub:02d}')
        np.save(f'cache/thingseeg2_preproc/transfer/sub-{sub:02d}/train_thingseeg2_{common_size}avg_transferred_from{transfer_source}.npy', eeg_transfered)
        np.save(f'cache/thingseeg2_preproc/transfer/sub-{sub:02d}/train_thingseeg2_{common_size}avg_nontransferred_from{transfer_source}.npy', eeg_nontransfered)

    elif transfer_source == 'other':
        eeg_common = np.load(f'data/thingseeg2_preproc/sub-{sub:02d}/train_thingseeg2_avg.npy')
        if timebins:
            eeg_common = eeg_common.transpose(0,2,1)
        original_shape = eeg_common.shape
        eeg_common = eeg_common[:common_size]
        eeg_common = eeg_common.reshape(eeg_common.shape[0],-1)
        other_subjects = [i for i in range(1, 11) if i != sub]
        eeg_new_preds = []
        for other_sub in other_subjects:
            eeg_transfer_from = np.load(f'data/thingseeg2_preproc/sub-{other_sub:02d}/train_thingseeg2_avg.npy')
            eeg_common_transfer_from = eeg_transfer_from[:common_size]
            if timebins:
                eeg_common_transfer_from = eeg_common_transfer_from.transpose(0,2,1)
            eeg_common_transfer_from = eeg_common_transfer_from.reshape(eeg_common_transfer_from.shape[0],-1)
            eeg_new_transfer_from = eeg_transfer_from[common_size:]
            if timebins:
                eeg_new_transfer_from = eeg_new_transfer_from.transpose(0,2,1)
            eeg_new_transfer_from = eeg_new_transfer_from.reshape(eeg_new_transfer_from.shape[0],-1)

            print(eeg_common.shape, eeg_new_transfer_from.shape)

            if timebins:
                eeg_new_pred = np.zeros(eeg_new_transfer_from.shape)
                print(f"Training Subject {other_sub} Augmentation Regression")
                for i in range(8):
                    reg = skl.Ridge(alpha=1000, max_iter=50000, fit_intercept=True)
                    reg.fit(eeg_common_transfer_from[:,i*170:(i+1)*170], eeg_common[:,i*170:(i+1)*170])
                    eeg_new_pred[:,i*170:(i+1)*170] = reg.predict(eeg_new_transfer_from[:,i*170:(i+1)*170])
                print('Augmentation training complete')
            else:
                print(f"Training Subject {other_sub} Augmentation Regression")
                reg = skl.Ridge(alpha=1000, max_iter=50000, fit_intercept=True)
                reg.fit(eeg_common_transfer_from, eeg_common)
                print('Augmentation training complete')
                weights_save_dir = f'cache/thingseeg2_preproc/transfer_regression_weights/sub-{sub:02d}/'
                weights_filename = f'regress_weights_{common_size}avg_transfered_from{other_sub}.pkl'
                if not os.path.exists(weights_save_dir):
                    os.makedirs(weights_save_dir)
                if saving_weights:
                    datadict = {
                        'weight' : reg.coef_,
                        'bias' : reg.intercept_,
                    }
                    with open(weights_save_dir + weights_filename, "wb") as f:
                        pickle.dump(datadict,f)

                eeg_new_pred = reg.predict(eeg_new_transfer_from)
            # eeg_transfered = np.concatenate((eeg_common, eeg_new_pred), axis=0)
            # eeg_transfered = eeg_transfered.reshape(original_shape)
            eeg_nontransfered = np.concatenate((eeg_common, eeg_new_transfer_from), axis=0)
            eeg_nontransfered = eeg_nontransfered.reshape(original_shape)
            if timebins:
                eeg_nontransfered = eeg_nontransfered.transpose(0,2,1)
            eeg_new_preds.append(eeg_new_pred)
            if not os.path.exists(f'cache/thingseeg2_preproc/transfer/sub-{sub:02d}'):
                os.makedirs(f'cache/thingseeg2_preproc/transfer/sub-{sub:02d}')
                np.save(f'cache/thingseeg2_preproc/transfer/sub-{sub:02d}/train_thingseeg2_{common_size}avg_nontransferred_from{other_sub}.npy', eeg_nontransfered)
        eeg_new_preds = np.stack(eeg_new_preds)
        eeg_new_pred = eeg_new_preds.mean(0)
        eeg_transfered = np.concatenate((eeg_common, eeg_new_pred), axis=0)
        eeg_transfered = eeg_transfered.reshape(original_shape)
        if timebins:
            eeg_transfered = eeg_transfered.transpose(0,2,1)
        np.save(f'cache/thingseeg2_preproc/transfer/sub-{sub:02d}/train_thingseeg2_{common_size}avg_transferred_from{transfer_source}.npy', eeg_transfered)
    

elif method == 'transfer':
    eeg_common = np.load(f'data/thingseeg2_preproc/sub-{sub:02d}/train_thingseeg2_avg.npy')
    eeg_common = eeg_common[:common_size]
    eeg_common = eeg_common.reshape(eeg_common.shape[0],-1)
    eeg_common_transfer_to = np.load(f'data/thingseeg2_preproc/sub-{transfer_source:02d}/train_thingseeg2_avg.npy')[:common_size]
    eeg_common_transfer_to = eeg_common_transfer_to.reshape(eeg_common_transfer_to.shape[0],-1)
    eeg_new = np.load(f'data/thingseeg2_preproc/sub-{sub:02d}/test_thingseeg2_avg.npy')
    eeg_new_original_shape = eeg_new.shape
    eeg_new = eeg_new.reshape(eeg_new.shape[0],-1)

    print(eeg_common.shape, eeg_new.shape)

    print("Training Transfer Regression")
    reg = skl.Ridge(alpha=1000, max_iter=50000, fit_intercept=True)
    reg.fit(eeg_common, eeg_common_transfer_to)
    print('Transfer training complete')

    eeg_new_pred = reg.predict(eeg_new)
    eeg_transfered = eeg_new_pred.reshape(eeg_new_original_shape)
    eeg_nontransfered = eeg_new.reshape(eeg_new_original_shape)
    if not os.path.exists(f'cache/thingseeg2_preproc/transfer/sub-{sub:02d}'):
        os.makedirs(f'cache/thingseeg2_preproc/transfer/sub-{sub:02d}')
    np.save(f'cache/thingseeg2_preproc/transfer/sub-{sub:02d}/test_thingseeg2_{common_size}avg_transferred_to{transfer_source}.npy', eeg_transfered)
    np.save(f'cache/thingseeg2_preproc/transfer/sub-{sub:02d}/test_thingseeg2_avg_nontransferred_to{transfer_source}.npy', eeg_nontransfered)