import numpy as np

# https://docs.scipy.org/doc/numpy/reference/generated/numpy.hanning.html
# https://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html
brain_freq_bands = {'all': (1, 45),
                    'delta': (1, 4),
                    'theta': (4, 8),
                    'alpha': (8, 12),
                    'beta': (12, 30),
                    'gamma': (30, 45)
                    }

WINDOW_SIZE = 12
WINDOW_FUNCTION = np.hanning(WINDOW_SIZE)


def apply_window_function(signal, window_function):
    return signal * window_function


def fourier_transform(windowed_signal):
    return np.fft.fft(windowed_signal)


def avg_band_amplitude(frequencies, lower_limit, upper_limit):
    frequency_band = frequencies[(frequencies >= lower_limit) * (frequencies <= upper_limit)]
    return np.mean(np.absolute(frequency_band))


def extract_amplitudes(input_signal):
    windowed_signal = apply_window_function(input_signal, WINDOW_FUNCTION)
    frequency_spectrum = fourier_transform(windowed_signal)
    amplitudes = []
    for wave, band_range in brain_freq_bands.items():
        amplitudes.append(avg_band_amplitude(frequency_spectrum, band_range[0], band_range[1]))
    return amplitudes


'''
def main():
    time = np.linspace(0, 0.5, 500)
    input_signal = np.sin(40 * 2 * np.pi * time) + 0.5 * np.sin(90 * 2 * np.pi * time)
    extract_amplitudes(input_signal)


if __name__ == "__main__":
    main()
'''
