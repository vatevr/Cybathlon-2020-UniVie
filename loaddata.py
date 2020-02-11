import numpy as np
import time
from scipy import io as sio
from random import random as rand

def loadeeg():
    fs = 512  

    update_time = 0.01

    nsample = np.int(fs*update_time)
    data = sio.loadmat('data/s01.mat', squeeze_me=True, struct_as_record=False, verify_compressed_data_integrity=False)['eeg']

    imagery_left = data.imagery_left - \
        data.imagery_left.mean(axis=1, keepdims=True)
    imagery_right = data.imagery_right - \
        data.imagery_right.mean(axis=1, keepdims=True)

    eeg_data_l = np.vstack([imagery_left * 1e-6, data.imagery_event])
    eeg_data_r = np.vstack([imagery_right * 1e-6,
                            data.imagery_event * 2])
    eeg_data = np.hstack([eeg_data_l, np.zeros((eeg_data_l.shape[0], 500)),
                          eeg_data_r])

    return eeg_data