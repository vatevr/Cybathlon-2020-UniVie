import sys
import time
from topomap_plot import plot_single_topomap
from mne.channels.layout import _auto_topomap_coords as pos_from_raw
import mne
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import seaborn as sea

brain_freq_bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'gamma': (30, 45)
}


# Calculates an avg of the power within the given indexes
def avg_band_amplitude(power, lower_limit_index, upper_limit_index):
    range = power[:, lower_limit_index:upper_limit_index]
    return np.mean(range, axis=1)


# Returns for each brain wave bandwidth the average amplitude within that bandwidth for each electrode
def extract_amplitudes(data, sampling_rate):
    frequencies, power = calculate_psd(data, sampling_rate)
    rescaled_power = 10 * np.log10(power)
    amplitudes = []
    for wave, band_range in brain_freq_bands.items():
        lower_index = next(index for index, value in enumerate(frequencies) if value > band_range[0])
        upper_index = next(index for index, value in enumerate(frequencies) if value > band_range[1])
        amplitudes.append(avg_band_amplitude(rescaled_power, lower_index, upper_index))
    return amplitudes


def calculate_psd(input_signal, sampling_rate):
    return scipy.signal.welch(x=input_signal, fs=sampling_rate)


def main():
    # Subject 4
    raw_s4 = mne.io.Raw('../data/S4_4chns.raw', preload=True)
    events_from_annot, event_dict = mne.events_from_annotations(raw_s4)
    epochs_s4 = mne.Epochs(raw=raw_s4, events=events_from_annot, event_id=20)

    # Subject 2
    raw_s2 = mne.io.Raw('../data/S2_4chns.raw', preload=True)
    events_from_annot, event_dict = mne.events_from_annotations(raw_s2)
    epochs_s2 = mne.Epochs(raw=raw_s2, events=events_from_annot, event_id=20)

    sampling_rate = 500.

    avg_amplitudes_per_epoch_s2 = []
    for epoch in epochs_s2:
        avg_amplitudes_per_epoch_s2.append(extract_amplitudes(epoch, sampling_rate))

    avg_amplitudes_per_epoch_s4 = []
    for epoch in epochs_s4:
        avg_amplitudes_per_epoch_s4.append(extract_amplitudes(epoch, sampling_rate))

    result_s2 = np.array(avg_amplitudes_per_epoch_s2)
    result_s4 = np.array(avg_amplitudes_per_epoch_s4)

    # Correlate all of the events, and each band from one subject with the same one from the other, with each channel with the same one
    corr = []
    '''for band in range(5):
        for channel in range(4):
            corr.append(np.corrcoef(x=result_s2[:, band, channel], y=result_s4[:, band, channel])[0][1])
    '''
    for band1 in range(5):
        for channel1 in range(4):
            corr1 = []
            for band2 in range(5):
                for channel2 in range(4):
                    corr1.append(np.corrcoef(x=result_s2[:, band1, channel1], y=result_s4[:, band2, channel2])[0][1])
            corr.append(corr1)

    ax = sea.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sea.diverging_palette(20, 220, n=200),
        square=True
    )

    plt.savefig('eval.png')


if __name__ == "__main__":
    main()
