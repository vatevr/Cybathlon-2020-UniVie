import numpy as np
from scipy import signal
import scipy


class AmplitudeExtractionBox(OVBox):

    def apply_window_function(self, signal, window_function):
        self.window_function = scipy.signal.hann(M=int(self.window_size * self.sampling_freq), sym=False)
        return signal * window_function

    def frequency_spectrum(self, windowed_signal):
        return 10 * np.log10(np.absolute(np.fft.fft(windowed_signal)))

    def avg_band_amplitude(self, spectrum, lower_limit, upper_limit):
        frequency_band = spectrum[:, int(lower_limit / self.freq_resolution):int(upper_limit / self.freq_resolution)]
        return np.mean(frequency_band, axis=1)

    def extract_amplitudes(self, input_signal):
        windowed_signal = self.apply_window_function(input_signal, self.window_function)
        spectrum = self.frequency_spectrum(windowed_signal)
        amplitudes = []
        for wave, band_range in self.bands.items():
            band_amplitude = self.avg_band_amplitude(spectrum, int(band_range[0]), int(band_range[1]))
            if any(np.isnan(band_amplitude)):
                for index, channel_amplitude in enumerate(band_amplitude):
                    if np.isnan(channel_amplitude):
                        band_amplitude[index] = 0.
                amplitudes.append(band_amplitude)
                #raise ValueError("No frequencies exist in this bandwidth")
            else:
                amplitudes.append(band_amplitude)
        return amplitudes

    def __init__(self):
        OVBox.__init__(self)
        self.startTime = 0.
        self.channelCount = 0
        self.endTime = 0.
        self.dimensionSizes = list()
        self.dimensionLabels = list()
        self.timeBuffer = list()
        self.signalBuffer = None
        self.signalHeader = None
        self.window_size = 0
        self.sampling_freq = 500
        self.freq_resolution = None
        self.window_function = None
        self.bands = None

    def initialize(self):
        self.sampling_freq = int(self.setting['Sampling frequency'])
        self.channelCount = int(self.setting['Channel count'])
        self.window_size = int(self.setting['Window Size'])
        self.freq_resolution = 1. / self.window_size
        delta = list(self.setting['delta'].split("-"))
        theta = list(self.setting['theta'].split("-"))
        alpha = list(self.setting['alpha'].split("-"))
        beta = list(self.setting['beta'].split("-"))
        gamma = list(self.setting['gamma'].split("-"))
        self.bands = {'delta': delta, 'theta': theta, 'alpha': alpha, 'beta': beta, 'gamma': gamma}

    def process(self):
        for chunkIndex in range(len(self.input[0])):
            if (type(self.input[0][chunkIndex]) == OVSignalHeader):
                self.signalHeader = self.input[0].pop()
                outputHeader = OVSignalHeader(
                    self.signalHeader.startTime,
                    self.signalHeader.endTime,
                    [len(self.bands), self.channelCount],
                    ['delta', 'theta', 'alpha', 'beta', 'gamma'],
                    self.signalHeader.samplingRate)

                self.output[0].append(outputHeader)


            elif (type(self.input[0][chunkIndex]) == OVSignalBuffer):
                chunk = self.input[0].pop()
                numpyBuffer = np.array(chunk).reshape(tuple(self.signalHeader.dimensionSizes))
                numpyBuffer = self.extract_amplitudes(numpyBuffer)
                if numpyBuffer is None:
                    raise ValueError("numpyBuffer is none")
                chunk = OVSignalBuffer(chunk.startTime, chunk.endTime, numpyBuffer)
                self.output[0].append(chunk)


            elif (type(self.input[0][chunkIndex]) == OVSignalEnd):
                self.output[0].append(self.input[0].pop())


box = AmplitudeExtractionBox()