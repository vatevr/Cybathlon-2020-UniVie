import mne
import numpy as np
from amplitude_extraction_evaluation import extract_amplitudes
from amplitude_extraction_welch_evaluation import extract_amplitudes as extract_amplitudes_welch
from amplitude_extraction_multitaper_evaluation import extract_amplitudes as extract_amplitudes_multitaper
from topomap_plot import plot_single_topomap
from mne.channels.layout import _auto_topomap_coords as pos_from_raw


def main():
    # Subject 4
    raw = mne.io.Raw('../data/S4_4chns.raw', preload=True)
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage)
    events_from_annot, event_dict = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw=raw, events=events_from_annot, event_id=[10, 20])
    pos = pos_from_raw(raw.info, None)
    sampling_rate = 500.
    beta = 3

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
        channel_corr_coeff_fft.append((np.corrcoef(labels, results_fft[:, beta, channel])[0][1]**2)*100)
        channel_corr_coeff_welch.append((np.corrcoef(labels, results_welch[:, beta, channel])[0][1]**2)*100)
        channel_corr_coeff_multitaper.append((np.corrcoef(labels, results_multitaper[:, beta, channel])[0][1]**2)*100)

    print(channel_corr_coeff_fft)

    # Plotting
    plot_single_topomap(channel_corr_coeff_fft, pos, title='', cmap_rb=True)
    plot_single_topomap(channel_corr_coeff_welch, pos, title='', cmap_rb=True)
    plot_single_topomap(channel_corr_coeff_multitaper, pos, title='', cmap_rb=True)


if __name__ == "__main__":
    main()
