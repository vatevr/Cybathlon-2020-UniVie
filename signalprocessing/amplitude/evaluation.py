import mne
import numpy as np
from amplitude_extraction_evaluation import extract_amplitudes
from amplitude_extraction_welch_evaluation import extract_amplitudes as extract_amplitudes_welch
from amplitude_extraction_multitaper_evaluation import extract_amplitudes as extract_amplitudes_multitaper
from topomap_plot import plot_single_topomap
from mne.channels.layout import _auto_topomap_coords as pos_from_raw
import matplotlib.pyplot as plt

from utils import load_epochs_from_raw, load_data


def main():
    sampling_rate = 500.
    alpha = 2
    number_channels = 122
    bads_TF = ['F2', 'FFC2h', 'POO10h', 'O2']

    # Loading data -> 4chns
    raw = mne.io.Raw('../data/S4_4chns.raw', preload=True)
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage)
    events_from_annot, event_dict = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw=raw, events=events_from_annot, event_id=[10, 20])
    pos = pos_from_raw(raw.info, None)

    # Preprocessing and loading of data -> TF
    data, raw, pos, picks = load_data('../data/20191201_Cybathlon_TF_Session1_Block1.vhdr',
                                      bads_TF, None)
    data2, raw2, pos2, picks2 = load_data('../data/20191201_Cybathlon_TF_Session1_Block2.vhdr',
                                          bads_TF, None)
    epochs1 = load_epochs_from_raw(raw=raw, picks=picks, events=[1, 2], times=[1, 6])
    epochs2 = load_epochs_from_raw(raw=raw2, picks=picks, events=[1, 2], times=[1, 6])
    epochs = mne.concatenate_epochs([epochs1, epochs2])

    # Preprocessing and loading of data -> SZ
    # data, raw, pos, picks = load_data('../data/20191210_Cybathlon_SAZ_Session1.vhdr',
    #                                   [], None)
    # epochs = load_epochs_from_raw(raw=raw, picks=picks, events=[1, 2], times=[1, 6])

    number_events = len(epochs.events)

    labels = np.zeros(len(epochs.events))
    for i in range(len(epochs.events)):
        for key in epochs[i].event_id:
            labels[i] = key

    avg_amplitudes_per_epoch_fft = []
    avg_amplitudes_per_epoch_welch = []
    avg_amplitudes_per_epoch_multitaper = []

    # Calculate average amplitudes
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

    # Calculate corr-coeff
    channel_corr_coeff_fft = []
    channel_corr_coeff_welch = []
    channel_corr_coeff_multitaper = []
    for channel in range(number_channels):
        channel_corr_coeff_fft.append((np.corrcoef(labels, results_fft[:, alpha, channel])[0][1] ** 2))
        channel_corr_coeff_welch.append((np.corrcoef(labels, results_welch[:, alpha, channel])[0][1] ** 2))
        channel_corr_coeff_multitaper.append(
            (np.corrcoef(labels, results_multitaper[:, alpha, channel])[0][1] ** 2))

    # # Plotting
    plot_single_topomap(channel_corr_coeff_fft, pos, title='FFT - TF', cmap_rb=True)
    plot_single_topomap(channel_corr_coeff_welch, pos, title='Welch - TF', cmap_rb=True)
    plot_single_topomap(channel_corr_coeff_multitaper, pos, title='Multitaper - TF', cmap_rb=True)

    # Scatterplot
    right_feature = []
    left_feature = []
    c3 = 1
    c4 = 2
    #
    # for i in range(number_events):
    #     if labels[i] == 1:
    #         left_feature.append(results_fft[i, alpha, c4])
    #     else:
    #         right_feature.append(results_fft[i, alpha, c4])

    # fig = plt.figure()
    # ax.scatter(labels, results_fft[:, alpha, 1], color='r')
    # ax.set_xlabel('Labels')
    # ax.set_ylabel('Features')
    # ax.set_title('FFT')
    # plt.show()
    plt.xlabel('Events')
    plt.ylabel('Features')
    # plt.scatter(range(int(number_events/2)), right_feature)
    # plt.scatter(range(int(number_events/2)), left_feature)
    # plt.scatter(labels, results_fft[:, alpha, c4])
    # plt.show()


if __name__ == "__main__":
    main()
