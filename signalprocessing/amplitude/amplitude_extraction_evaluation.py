import numpy as np
import sys
from topomap_plot import plot_single_topomap
from mne.channels.layout import _auto_topomap_coords as pos_from_raw
import time
import scipy.signal
import mne
import matplotlib.pyplot as plt
import seaborn as sea

from utils import load_epochs_from_path

brain_freq_bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'gamma': (30, 45)
}

SAMPLING_RATE = 500.


def apply_window_function(signal, window_size, sampling_rate):
    return signal * scipy.signal.hann(M=int(window_size * sampling_rate), sym=False)


def frequency_spectrum(windowed_signal):
    return 10 * np.log10(np.absolute(np.fft.fft(windowed_signal)))


# pass whole spectrum for all channels to this function
def avg_band_amplitude(spectrum, lower_limit, upper_limit, freq_resolution):
    # retrieve a frequency band across all channels, by using the scaling determined by the frequency resolution
    frequency_band = spectrum[:, int(lower_limit / freq_resolution):int(upper_limit / freq_resolution)]
    return np.mean(frequency_band, axis=1)


# Returns for each brain wave bandwidth the average amplitude within that bandwidth for each electrode
def extract_amplitudes(input_signal, window_size, sampling_rate):
    windowed_signal = apply_window_function(input_signal, window_size, sampling_rate)
    spectrum = frequency_spectrum(windowed_signal)
    amplitudes = []
    for wave, band_range in brain_freq_bands.items():
        amplitudes.append(avg_band_amplitude(spectrum, band_range[0], band_range[1], 1. / window_size))
    return amplitudes


def avg_amplitudes_per_epochs(epochs):
    avg_amplitudes_per_epoch = []
    for epoch in epochs:
        samples_per_epoch = epoch.shape[1]
        window_size = samples_per_epoch / SAMPLING_RATE
        avg_amplitudes_per_epoch.append(extract_amplitudes(epoch, window_size, SAMPLING_RATE))
    return avg_amplitudes_per_epoch


def main():
    avg_amplitudes_per_epoch_s2 = avg_amplitudes_per_epochs(load_epochs_from_path(path='../data/S2_4chns.raw', events=20))
    avg_amplitudes_per_epoch_s4 = avg_amplitudes_per_epochs(load_epochs_from_path(path='../data/S4_4chns.raw', events=20))

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
                    corr1.append((np.corrcoef(x=result_s2[:, band1, channel1], y=result_s4[:, band2, channel2])[0][
                                      1] ** 2) * 100)
            corr.append(corr1)

    ax = sea.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sea.diverging_palette(20, 220, n=200),
        square=True
    )

    plt.show()


if __name__ == "__main__":
    main()
