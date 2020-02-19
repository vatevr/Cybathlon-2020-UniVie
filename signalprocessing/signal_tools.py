import numpy as np
import scipy

brain_freq_bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'gamma': (30, 45)
}

# Calculates an avg of the power within the given indexes
def avg_band_amplitude(power, lower_limit_index, upper_limit_index):
    range = power[:, lower_limit_index:upper_limit_index]
    return np.mean(range, axis=1)

def calculate_psd(input_signal, sampling_rate):
    return scipy.signal.welch(x=input_signal, fs=sampling_rate)

# Returns for each brain wave bandwidth the average amplitude within that bandwidth for each electrode
def extract_amplitudes_welch(data, sampling_rate):
    data = data.T
    frequencies, power = calculate_psd(data, sampling_rate)
    rescaled_power = 10 * np.log10(power)
    amplitudes = []
    for wave, band_range in brain_freq_bands.items():
        lower_index = next(index for index, value in enumerate(frequencies) if value > band_range[0])
        upper_index = next(index for index, value in enumerate(frequencies) if value > band_range[1])
        amplitudes.append(avg_band_amplitude(rescaled_power, lower_index, upper_index))
    return np.array(amplitudes)