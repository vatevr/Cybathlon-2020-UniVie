import matplotlib as plt
import numpy as np

eeg_freq_bands = {'all': (1, 45),
                  'delta': (1, 4),
                  'theta': (4, 8),
                  'alpha': (8, 12),
                  'beta': (12, 30),
                  'gamma': (30, 45)
                  }

time = np.linspace(0, 0.5, 500)
input_signal = np.sin(40 * 2 * np.pi * time) + 0.5 * np.sin(90 * 2 * np.pi * time)
#create a matrix from this

def apply_window_function(signal, window_function):
    return 0


def fourier_transform(signal):
    return np.fft.fft(signal)


def amplitude_of_band(frequencies, lower_limit, upper_limit):
    frequency_band = np.logical_and(frequencies >= lower_limit, frequencies <= upper_limit)
    return 0


frequency_spectrum = fourier_transform(input_signal)
