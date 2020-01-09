import numpy as np
import sys
from topomap_plot import plot_single_topomap
from mne.channels.layout import _auto_topomap_coords as pos_from_raw
import time
import scipy.signal
import mne
import matplotlib.pyplot as plt

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
    # retrieve a frequency band across all channels, by using the scaling determined by the frequency resolution
    frequency_band = spectrum[:, int(lower_limit / FREQ_RESOLUTION):int(upper_limit / FREQ_RESOLUTION)]
    return np.mean(frequency_band, axis=1)


# Returns for each brain wave bandwidth the average amplitude within that bandwidth for each electrode
def extract_amplitudes(input_signal):
    windowed_signal = apply_window_function(input_signal, WINDOW_FUNCTION)
    spectrum = frequency_spectrum(windowed_signal)
    amplitudes = []
    for wave, band_range in brain_freq_bands.items():
        amplitudes.append(avg_band_amplitude(spectrum, band_range[0], band_range[1]))
    return amplitudes


def main():
    print(WINDOW_SIZE, SAMPLING_RATE)

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

    # Plotting
    # plt.plot(amplitudes[2])
    # plt.show()
    # print(raw.info.ch_names)
    plot_single_topomap(amplitudes[2], pos, title='', cmap_rb=True)
    print("elapsed time:", end - start)


if __name__ == "__main__":
    main()
