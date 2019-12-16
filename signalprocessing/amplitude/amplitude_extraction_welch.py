import numpy as np
import sys
import matplotlib.pyplot as plt
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


# pass whole spectrum for all channels to this function
def avg_band_amplitude(spectrum, lower_limit, upper_limit):
    frequency_band = spectrum[:, int(lower_limit / FREQ_RESOLUTION):int(upper_limit / FREQ_RESOLUTION)]
    return np.mean(frequency_band, axis=1)


def calculate_psd(input_signal):
    return scipy.signal.welch(x=input_signal, fs=SAMPLING_RATE)


def main():
    print(WINDOW_SIZE, SAMPLING_RATE)
    raw = mne.io.read_raw_brainvision('../data/20191104_Cybathlon_Test_1.vhdr')
    t_idx = raw.time_as_index([100., 110.])
    data, times = raw[:, t_idx[0]:t_idx[1]]
    start = time.time()
    f, pxx_den = calculate_psd(data)
    print(pxx_den)
    plt.semilogy(f, pxx_den.T)
    #plt.ylim([0.5e-3, 1])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()
    end = time.time()
    # print(raw.info.ch_names)
    # plot_data_for_single_channel(amplitudes[2], raw)
    print("elapsed time:", end - start)


if __name__ == "__main__":
    main()
