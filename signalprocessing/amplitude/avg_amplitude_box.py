import numpy as np
import scipy


class AmplitudeExtractionBox(OVBox):

    def apply_window_function(self, signal, window_function):
        return signal * window_function

    def frequency_spectrum(self, windowed_signal):
        return 10 * np.log10(np.absolute(np.fft.fft(windowed_signal)))

        # pass whole spectrum for all channels to this function

    def avg_band_amplitude(self, spectrum, lower_limit, upper_limit):
        frequency_band = spectrum[:, int(lower_limit / self.freq_resolution):int(upper_limit / self.freq_resolution)]
        return np.mean(frequency_band, axis=1)

    def extract_amplitudes(self, input_signal):
        windowed_signal = self.apply_window_function(input_signal, self.window_function)
        spectrum = self.frequency_spectrum(windowed_signal)
        amplitudes = []
        for wave, band_range in self.brain_freq_bands.items():
            amplitudes.append(self.avg_band_amplitude(spectrum, band_range[0], band_range[1]))
        return amplitudes

    def __init__(self):
        OVBox.__init__(self)
        self.startTime = 0.
        self.endTime = 0.
        self.dimensionSizes = list()
        self.dimensionLabels = list()
        self.timeBuffer = list()
        self.signalBuffer = None
        self.signalHeader = None
        self.window_size = 10
        self.sampling_rate = 500
        self.freq_resolution = None
        self.window_function = None
        self.brain_freq_bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30),
                                 'gamma': (30, 45)}

    def initialize(self):
        self.sampling_rate = int(self.setting['Sampling frequency'])
        self.freq_resolution = 1. / self.window_size
        self.window_function = scipy.signal.hann(M=int(self.window_size * self.sampling_rate), sym=False)

    def process(self):
        for chunkIndex in range(len(self.input[0])):
            if (type(self.input[0][chunkIndex]) == OVSignalHeader):
                self.signalHeader = self.input[0].pop()

                outputHeader = OVSignalHeader(
                    self.signalHeader.startTime,
                    self.signalHeader.endTime,
                    [5, self.channelCount],
                    ['delta', 'theta', 'alpha', 'beta', 'gamma'],
                    self.signalHeader.samplingRate)

                self.output[0].append(outputHeader)


            elif (type(self.input[0][chunkIndex]) == OVSignalBuffer):
                chunk = self.input[0].pop()
                numpyBuffer = np.array(chunk).reshape(tuple(self.signalHeader.dimensionSizes))
                numpyBuffer = self.extract_amplitudes(self, numpyBuffer)
                chunk = OVSignalBuffer(chunk.startTime, chunk.endTime, numpyBuffer.tolist())
                self.output[0].append(chunk)


            elif (type(self.input[0][chunkIndex]) == OVSignalEnd):
                self.output[0].append(self.input[0].pop())


box = AmplitudeExtractionBox()
