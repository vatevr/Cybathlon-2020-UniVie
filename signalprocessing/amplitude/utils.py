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
    # Remove bad channels from analysis
    print(raw.ch_names)
    raw.info['bads'] = bads
    picks = mne.pick_types(raw.info, eeg=True, stim=False, exclude='bads')
    pos = pos_from_raw(raw.info, picks)
    if time_index is not None:
        t_idx = raw.time_as_index(time_index)
        data = raw.get_data(picks, start=t_idx[0], stop=t_idx[1])
    else:
        data = raw.get_data(picks)
    return data, raw, pos, picks


def load_epochs_from_path(path, events):
    raw = mne.io.Raw(path, preload=True)
    events_from_annot, event_dict = mne.events_from_annotations(raw)
    if not events:
        return mne.Epochs(raw=raw, events=events_from_annot)
    return mne.Epochs(raw=raw, events=events_from_annot, event_id=events)


def load_epochs_from_raw(raw, picks, events, times):
    events_from_annot, event_dict = mne.events_from_annotations(raw)
    return mne.Epochs(raw, events_from_annot, baseline=None, tmin=times[0], tmax=times[1], picks=picks, event_id=events)
