import sys
import time
from topomap_plot import plot_single_topomap
from mne.channels.layout import _auto_topomap_coords as pos_from_raw
import mne
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

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
    plt.plot(power)
    plt.show()
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
    raw = mne.io.read_raw_brainvision('../data/20191201_Cybathlon_TF_Session1_RS.vhdr', preload=True)
    raw.set_eeg_reference(ref_channels='average')
    raw.rename_channels({'O9': 'I1', 'O10': 'I2'})
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage)
    raw.rename_channels({'I1': 'O9', 'I2': 'O10'})
    t_idx = raw.time_as_index([100., 110.])
    # Remove bad channels from analysis
    raw.info['bads'] = ['F2', 'FFC2h', 'POO10h', 'O2']
    picks = mne.pick_types(raw.info, eeg=True, stim=False, exclude='bads')
    data = raw.get_data(picks, start=t_idx[0], stop=t_idx[1])
    pos = pos_from_raw(raw.info, picks)

    # Calculations
    start = time.time()
    amplitudes = extract_amplitudes(data)
    end = time.time()
    print("elapsed time:", end - start)

    # Plotting
    # plt.semilogy(frequencies, power.T)
    # plt.xlabel('Frequency')
    # plt.ylabel('Power')
    # plt.show()
    # plt.plot(amplitudes[2])
    # plt.show()
    # plot_single_topomap(amplitudes[2], pos, title='', cmap_rb=True)


if __name__ == "__main__":
    main()
