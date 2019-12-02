import numpy as np
import scipy.signal
import mne
import matplotlib.pyplot as plt

# https://docs.scipy.org/doc/numpy/reference/generated/numpy.hanning.html
# https://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html
brain_freq_bands = {
                    'delta': (1, 4),
                    'theta': (4, 8),
                    'alpha': (8, 12),
                    'beta': (12, 30),
                    'gamma': (30, 45)
                    }

WINDOW_SIZE = 2.
SAMPLING_RATE = 500
FREQ_RESOLUTION = 1. / WINDOW_SIZE
WINDOW_FUNCTION = scipy.signal.hann(M=int(WINDOW_SIZE * SAMPLING_RATE), sym=False)


def apply_window_function(signal, window_function):
    # build a matrix for window function
    return signal * window_function


def fourier_transform(windowed_signal):
    return np.fft.fft(windowed_signal)


def avg_band_amplitude(spectrum, lower_limit, upper_limit):
    frequency_band = spectrum[int(lower_limit / FREQ_RESOLUTION):int(upper_limit / FREQ_RESOLUTION)]
    return np.mean(np.absolute(frequency_band))


def extract_amplitudes(input_signal):
    windowed_signal = apply_window_function(input_signal, WINDOW_FUNCTION)
    frequency_spectrum = fourier_transform(windowed_signal)
    plt.plot(10*np.log10(np.absolute(frequency_spectrum)))
    plt.show()
    amplitudes_all_channels = []
    for channel in frequency_spectrum.T[0:1]:
        amplitudes = []
        for wave, band_range in brain_freq_bands.items():
            amplitudes.append(avg_band_amplitude(channel, band_range[0], band_range[1]))
        amplitudes_all_channels.append(amplitudes)
    return amplitudes_all_channels


def main():
    raw = mne.io.read_raw_brainvision('../data/20191104_Cybathlon_Test_1.vhdr')
    # loads data from 0 to 1 sec
    t_idx = raw.time_as_index([0., 2.])
    data, times = raw[:, t_idx[0]:t_idx[1]]
    plt.plot(extract_amplitudes(data))
    plt.show()


# time = np.linspace(0, 0.5, 500)
# input_signal = np.sin(40 * 2 * np.pi * time) + 0.5 * np.sin(90 * 2 * np.pi * time)
# extract_amplitudes(input_signal)


if __name__ == "__main__":
    main()
