import mne
import numpy as np
from amplitude_extraction_evaluation import extract_amplitudes
from amplitude_extraction_welch_evaluation import extract_amplitudes as extract_amplitudes_welch
from amplitude_extraction_multitaper_evaluation import extract_amplitudes as extract_amplitudes_multitaper
from topomap_plot import plot_single_topomap
from mne.channels.layout import _auto_topomap_coords as pos_from_raw
import matplotlib.pyplot as plt


def main():
    # raw = mne.io.Raw('../data/S4_4chns.raw', preload=True)
    # montage = mne.channels.make_standard_montage('standard_1005')
    # raw.set_montage(montage)
    # events_from_annot, event_dict = mne.events_from_annotations(raw)
    # epochs = mne.Epochs(raw=raw, events=events_from_annot, event_id=[10, 20])
    # pos = pos_from_raw(raw.info, None)
    # sampling_rate = 500.
    # alpha = 2

    # Preprocessing and loading of data
    raw = mne.io.read_raw_brainvision('../data/20191201_Cybathlon_TF_Session1_Block1.vhdr', preload=True)
    raw2 = mne.io.read_raw_brainvision('../data/20191201_Cybathlon_TF_Session1_Block2.vhdr', preload=True)
    raw.set_eeg_reference(ref_channels='average')
    raw2.set_eeg_reference(ref_channels='average')
    raw.rename_channels({'O9': 'I1', 'O10': 'I2'})
    raw2.rename_channels({'O9': 'I1', 'O10': 'I2'})
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage)
    raw2.set_montage(montage)
    raw.rename_channels({'I1': 'O9', 'I2': 'O10'})
    raw2.rename_channels({'I1': 'O9', 'I2': 'O10'})
    t_idx = raw.time_as_index([100., 110.])
    # Remove bad channels from analysis
    raw.info['bads'] = ['F2', 'FFC2h', 'POO10h', 'O2']
    raw2.info['bads'] = ['F2', 'FFC2h', 'POO10h', 'O2']
    picks = mne.pick_types(raw.info, eeg=True, stim=False, exclude='bads')
    events_from_annot, event_dict = mne.events_from_annotations(raw)
    events_from_annot2, event_dict2 = mne.events_from_annotations(raw2)
    epochs1 = mne.Epochs(raw, events_from_annot, picks=picks, event_id=[1, 2])
    epochs2 = mne.Epochs(raw2, events_from_annot2, picks=picks, event_id=[1, 2])
    epochs = mne.concatenate_epochs([epochs1, epochs2])
    data = raw.get_data(picks, start=t_idx[0], stop=t_idx[1])
    pos = pos_from_raw(raw.info, picks)


    labels = np.zeros(len(epochs.events))
    for i in range(len(epochs.events)):
        for key in epochs[i].event_id:
            labels[i] = key

    avg_amplitudes_per_epoch_fft = []
    avg_amplitudes_per_epoch_welch = []
    avg_amplitudes_per_epoch_multitaper = []
    for epoch in epochs:
        samples_per_epoch = epoch.shape[1]
        window_size = samples_per_epoch / sampling_rate
        avg_amplitudes_per_epoch_fft.append(extract_amplitudes(epoch, window_size, sampling_rate))
        avg_amplitudes_per_epoch_welch.append(extract_amplitudes_welch(epoch, sampling_rate))

    for i in range(len(epochs.events)):
        power, frequencies = mne.time_frequency.psd_multitaper(inst=epochs[i])
        avg_amplitudes_per_epoch_multitaper.append(extract_amplitudes_multitaper(power[0], frequencies))

    results_fft = np.array(avg_amplitudes_per_epoch_fft)
    results_welch = np.array(avg_amplitudes_per_epoch_welch)
    results_multitaper = np.array(avg_amplitudes_per_epoch_multitaper)

    channel_corr_coeff_fft = []
    channel_corr_coeff_welch = []
    channel_corr_coeff_multitaper = []
    for channel in range(4):
        channel_corr_coeff_fft.append((np.corrcoef(labels, results_fft[:, alpha, channel])[0][1] ** 2) * 100)
        channel_corr_coeff_welch.append((np.corrcoef(labels, results_welch[:, alpha, channel])[0][1] ** 2) * 100)
        channel_corr_coeff_multitaper.append(
            (np.corrcoef(labels, results_multitaper[:, alpha, channel])[0][1] ** 2) * 100)

    # # Plotting
    plot_single_topomap(channel_corr_coeff_fft, pos, title='FFT', cmap_rb=True)
    plot_single_topomap(channel_corr_coeff_welch, pos, title='Welch', cmap_rb=True)
    plot_single_topomap(channel_corr_coeff_multitaper, pos, title='Multitaper', cmap_rb=True)

    # Scatterplot
    # right_feature = []
    # left_feature = []
    #
    # for i in range(300):
    #     if labels[i] == 20:
    #         left_feature.append(results_fft[i, alpha, 1])
    #     else:
    #         right_feature.append(results_fft[i, alpha, 1])

    # fig = plt.figure()
    # ax.scatter(labels, results_fft[:, alpha, 1], color='r')
    # ax.set_xlabel('Labels')
    # ax.set_ylabel('Features')
    # ax.set_title('FFT')
    # plt.show()
    # plt.xlabel('Events')
    # plt.ylabel('Features')
    # plt.scatter(range(150), right_feature)
    # plt.scatter(range(150), left_feature)
    # plt.scatter(labels, results_fft[:, alpha, 1])
    # plt.show()


if __name__ == "__main__":
    main()
