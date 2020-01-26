import mne
from mne.channels.layout import _auto_topomap_coords as pos_from_raw


# Returns raw and loaded data
def load_data(path, bads, time_index):
    raw = mne.io.read_raw_brainvision(path, preload=True)
    raw.set_eeg_reference(ref_channels='average')
    raw.rename_channels({'O9': 'I1', 'O10': 'I2'})
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage)
    raw.rename_channels({'I1': 'O9', 'I2': 'O10'})
    t_idx = raw.time_as_index([100., 110.])
    # Remove bad channels from analysis
    raw.info['bads'] = ['F2', 'FFC2h', 'POO10h', 'O2']
    picks = mne.pick_types(raw.info, eeg=True, stim=False, exclude='bads')
    data = raw.get_data(picks, start=t_idx[0], stop=t_idx[1])
    pos = pos_from_raw(raw.info, picks)
    return data, raw, pos, picks
