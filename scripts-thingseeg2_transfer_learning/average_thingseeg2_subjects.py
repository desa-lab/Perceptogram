import numpy as np
import os

for sub in range(1, 11):
    other_train_thingseeg2_avg = []
    other_subjects = [i for i in range(1, 11) if i != sub]
    print(other_subjects)
    for other_sub in other_subjects:
        data_dir = f'data/thingseeg2_preproc/sub-{other_sub:02d}/'
        train_thingseeg2_avg = np.load(data_dir + f'train_thingseeg2_avg.npy')
        other_train_thingseeg2_avg.append(train_thingseeg2_avg)
    other_train_thingseeg2_avg = np.stack(other_train_thingseeg2_avg)
    other_train_thingseeg2_avg = other_train_thingseeg2_avg.mean(0)
    data_dir = f'cache/thingseeg2_preproc/transfer/sub-{sub:02d}/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    np.save(data_dir + f'train_thingseeg2_avg_other.npy', other_train_thingseeg2_avg)