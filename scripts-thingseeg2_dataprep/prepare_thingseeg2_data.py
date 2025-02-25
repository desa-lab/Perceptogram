import numpy as np

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument('-avg', '--average', help='Number of averages', default=80)
args = parser.parse_args()
average=int(args.average)
if average != 80:
    param = f'{average}'
else:
    param = ''

for sub in range(1, 11):
    data_dir = f'data/thingseeg2_preproc/sub-{sub:02d}/'

    if average == 80:
        eeg_data_train = np.load(data_dir + 'preprocessed_eeg_training.npy', allow_pickle=True).item()
        print(f'\nTraining EEG data shape for sub-{sub:02d}:')
        print(eeg_data_train['preprocessed_eeg_data'].shape)
        print('(Training image conditions × Training EEG repetitions × EEG channels × '
            'EEG time points)')
        train_thingseeg2 = eeg_data_train['preprocessed_eeg_data'][:,:,:,20:]
        train_thingseeg2_avg = eeg_data_train['preprocessed_eeg_data'].mean(1)[:,:,20:]
        train_thingseeg2_avg_null = eeg_data_train['preprocessed_eeg_data'].mean(1)[:,:,:20]
        np.save(data_dir + 'train_thingseeg2.npy', train_thingseeg2)
        np.save(data_dir + 'train_thingseeg2_avg.npy', train_thingseeg2_avg)
        np.save(data_dir + 'train_thingseeg2_avg_null.npy', train_thingseeg2_avg_null)
    
    eeg_data_test = np.load(data_dir + 'preprocessed_eeg_test.npy', allow_pickle=True).item()
    print(f'\nTest EEG data shape for sub-{sub:02d}:')
    print(eeg_data_test['preprocessed_eeg_data'].shape)
    print('(Test image conditions × Test EEG repetitions × EEG channels × '
        'EEG time points)')
    test_thingseeg2 = eeg_data_test['preprocessed_eeg_data'][:,:,:,20:]
    test_thingseeg2_avg = eeg_data_test['preprocessed_eeg_data'][:,:average].mean(1)[:,:,20:]
    test_thingseeg2_avg_null = eeg_data_test['preprocessed_eeg_data'][:,:average].mean(1)[:,:,:20]
    np.save(data_dir + 'test_thingseeg2.npy', test_thingseeg2)
    np.save(data_dir + f'test_thingseeg2_avg{param}.npy', test_thingseeg2_avg)
    np.save(data_dir + f'test_thingseeg2_avg{param}_null.npy', test_thingseeg2_avg_null)
