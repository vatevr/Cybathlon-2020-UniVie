import sys
import time
from topomap_plot import plot_data_for_single_channel
import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy.signal

brain_freq_bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'gamma': (30, 45)
}

SAMPLING_RATE = int(sys.argv[1])


# Calculates an avg of the power within the given indexes
def avg_band_amplitude(power, lower_limit_index, upper_limit_index):
    range = power[:, lower_limit_index:upper_limit_index]
    return np.mean(range, axis=1)


# Returns for each brain wave bandwidth the average amplitude within that bandwidth for each electrode
def extract_amplitudes(data):
    frequencies, power = calculate_psd(data)
    rescaled_power = 10 * np.log10(power)
    amplitudes = []
    for wave, band_range in brain_freq_bands.items():
        lower_index = next(index for index, value in enumerate(frequencies) if value > band_range[0])
        upper_index = next(index for index, value in enumerate(frequencies) if value > band_range[1])
        amplitudes.append(avg_band_amplitude(rescaled_power, lower_index, upper_index))
    return amplitudes


def calculate_psd(input_signal):
    return scipy.signal.welch(x=input_signal, fs=SAMPLING_RATE)


def main():
    raw = mne.io.read_raw_brainvision('../data/20191201_Cybathlon_TF_Session1_RS.vhdr', preload=True)
    t_idx = raw.time_as_index([100., 110.])
    data, times = raw[:, t_idx[0]:t_idx[1]]
    start = time.time()
    amplitudes = extract_amplitudes(data)
    end = time.time()

    # plt.semilogy(frequencies, power.T)
    # plt.xlabel('Frequency')
    # plt.ylabel('Power')
    # plt.show()
    #plt.plot(amplitudes[2])
    # plt.show()
    plot_data_for_single_channel(amplitudes[2], raw)

    print("elapsed time:", end - start)


if __name__ == "__main__":
    main()
