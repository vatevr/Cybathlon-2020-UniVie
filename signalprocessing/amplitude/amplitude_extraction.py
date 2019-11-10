import matplotlib as plt
import numpy as np

brain_freq_bands = {'all': (1, 45),
                    'delta': (1, 4),
                    'theta': (4, 8),
                    'alpha': (8, 12),
                    'beta': (12, 30),
                    'gamma': (30, 45)
                    }

time = np.linspace(0, 0.5, 500)
input_signal = np.sin(40 * 2 * np.pi * time) + 0.5 * np.sin(90 * 2 * np.pi * time)


# create a matrix from this

def apply_window_function(signal):
    return 0


def fourier_transform(signal):
    return np.fft.fft(signal)


def avg_band_amplitude(frequencies, lower_limit, upper_limit):
    frequency_band = frequencies[(frequencies >= lower_limit) * (frequencies <= upper_limit)]
    return np.mean(np.absolute(frequency_band))


def extract_amplitudes():
    windowed_signal = apply_window_function(input_signal)
    frequency_spectrum = fourier_transform(windowed_signal)
    amplitudes = []
    for wave, band_range in brain_freq_bands.items():
        amplitudes.append(avg_band_amplitude(frequency_spectrum, band_range[0], band_range[1]))
    return amplitudes


def main():
    arr = np.array(range(-10, -20))
    frequs = (arr[(arr > 11) * (arr < 18)])
    print(np.mean(np.absolute(frequs)))
    # print(np.logical_and(arr[arr >= 12], arr[arr <= 18]))


if __name__ == "__main__":
    main()
