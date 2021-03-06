import sys
import time
#from topomap_plot import plot_single_topomap
from mne.channels.layout import _auto_topomap_coords as pos_from_raw
import mne
import numpy as np
import scipy.signal
#from utils import load_data
import matplotlib.pyplot as plt

brain_freq_bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'gamma': (30, 45)
}

SAMPLING_RATE = int(sys.argv[1])
ALPHA = 2


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
    # Preprocessing and loading of data
    data, raw, pos, picks = load_data('../data/20191201_Cybathlon_TF_Session1_RS.vhdr', ['F2', 'FFC2h', 'POO10h', 'O2'],
                                      [100., 110.])

    # Calculations
    start = time.time()
    amplitudes = extract_amplitudes(data)
    end = time.time()
    print("elapsed time:", end - start)

    # Plotting
    plot_single_topomap(amplitudes[ALPHA], pos, title='Welch - TF_Session1_RS - 10s', cmap_rb=True)


if __name__ == "__main__":
    main()
