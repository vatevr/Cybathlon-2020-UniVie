#Author: Melanie Balaz

import numpy as np
import sys
from optparse import OptionParser
from topomap_plot import plot_data_for_single_channel
import time
import scipy.signal
import mne

brain_freq_bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'gamma': (30, 45)
}

WINDOW_SIZE = float(sys.argv[1])
SAMPLING_RATE = int(sys.argv[2])
FREQ_RESOLUTION = 1. / WINDOW_SIZE
WINDOW_FUNCTION = scipy.signal.hann(M=int(WINDOW_SIZE * SAMPLING_RATE), sym=False)


def apply_window_function(signal, window_function):
    return signal * window_function


def frequency_spectrum(windowed_signal):
    return 10 * np.log10(np.absolute(np.fft.fft(windowed_signal)))


# pass whole spectrum for all channels to this function
def avg_band_amplitude(spectrum, lower_limit, upper_limit):
    frequency_band = spectrum[int(lower_limit / FREQ_RESOLUTION):int(upper_limit / FREQ_RESOLUTION)] #modified this line
    return np.mean(frequency_band, axis=0)                                                          #and this line


def extract_amplitudes(input_signal):
    windowed_signal = apply_window_function(input_signal, WINDOW_FUNCTION)
    spectrum = frequency_spectrum(windowed_signal)
    amplitudes = []
    for wave, band_range in brain_freq_bands.items():
        amplitudes.append(avg_band_amplitude(spectrum, band_range[0], band_range[1]))
    return amplitudes


def main():
    print(WINDOW_SIZE, SAMPLING_RATE)
    raw = mne.io.read_raw_brainvision('../data/20191104_Cybathlon_Test_1.vhdr')
    t_idx = raw.time_as_index([100., 110.])
    data, times = raw[:, t_idx[0]:t_idx[1]]
    start = time.time()
    # amplitudes = extract_amplitudes(data)
    end = time.time()
    # print(raw.info.ch_names)
    # plot_data_for_single_channel(amplitudes[2], raw)
    print("elapsed time:", end - start)


if __name__ == "__main__":
    main()