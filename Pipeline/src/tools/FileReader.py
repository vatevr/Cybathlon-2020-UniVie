import numpy as np
import mne
from scipy import io as sio


class FileReader():
    def __init__(self, datapath):
        self.datapath = datapath

    def load_mat(self):
        fs = 512

        update_time = 0.01

        nsample = np.int(fs * update_time)
        data = \
        sio.loadmat(self.datapath, squeeze_me=True, struct_as_record=False, verify_compressed_data_integrity=False)[
            'eeg']

        imagery_left = data.imagery_left - \
                       data.imagery_left.mean(axis=1, keepdims=True)
        imagery_right = data.imagery_right - \
                        data.imagery_right.mean(axis=1, keepdims=True)

        eeg_data_l = np.vstack([imagery_left, data.imagery_event])
        eeg_data_r = np.vstack([imagery_right,
                                data.imagery_event * 2])
        eeg_data = np.hstack([eeg_data_l, np.zeros((eeg_data_l.shape[0], 500)),
                              eeg_data_r])

        return eeg_data

    def load_CA(self):
        raw = mne.io.read_raw_brainvision(self.datapath, preload=True)
        # Set montage (location of channels)
        raw.rename_channels({'O9': 'I1', 'O10': 'I2'})
        montage = mne.channels.read_montage("standard_1005")
        raw.set_montage(montage)
        raw.rename_channels({'I1': 'O9', 'I2': 'O10'})
        # Remove bad channels from analysis
        raw.info['bads'] = ["PPO9h", "FFT7h", "P10", 'Pz']
        picks = mne.pick_types(raw.info, eeg=True, stim=False, exclude='bads')
        raw.set_eeg_reference(ref_channels="average")
        # Create events from triggers
        events = mne.events_from_annotations(raw)[0]
        # events = aggregate_events(events) # Aggregate
        tmin = 2  # time in seconds after trigger the trial should start
        tmax = tmin + 5  # time in seconds after trigger the trial should end
        epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, preload=True, baseline=None, picks=picks)
        return epochs, raw
