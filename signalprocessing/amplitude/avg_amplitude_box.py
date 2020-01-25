import numpy as np
import scipy.linalg
import scipy.io
import scipy.signal


class AmplitudeExtractionBox(OVBox):
    def __init__(self):
        OVBox.__init__(self)
        self.channelCount = 0
        self.samplingFrequency = 0
        self.windowSize = 10
        self.freqResolution = 0
        self.epochSampleCount = 0
        self.signalBuffer = list()
        self.windowFunction = None
        self.signalHeader = None

    def initialize(self):
        self.channelCount = int(self.setting['Channel count'])
        self.samplingFrequency = int(self.setting['Sampling frequency'])
        self.windowSize = int(self.setting['Generated epoch sample count'])
        self.freqResolution = 1. / self.windowSize

        delta = list(self.setting['delta'].split("-"))
        theta = list(self.setting['theta'].split("-"))
        alpha = list(self.setting['alpha'].split("-"))
        beta = list(self.setting['beta'].split("-"))
        gamma = list(self.setting['gamma'].split("-"))
        self.bands = {'delta': delta, 'theta': theta, 'alpha': alpha, 'beta': beta, 'gamma': gamma}

    def apply_window_function(self, signal, window_function):
        self.windowFunction = scipy.signal.hann(M=int(self.windowSize * self.samplingFrequency), sym=False)
        return signal * window_function

    def frequency_spectrum(self, windowed_signal):
        return 10 * np.log10(np.absolute(np.fft.fft(windowed_signal)))

    def avg_band_amplitude(self, spectrum, lower_limit, upper_limit):
        frequency_band = spectrum[:,
                         int(lower_limit / self.freqResolution):int(upper_limit / self.freqResolution)]
        return np.mean(frequency_band, axis=1)

    def do(self, input_signal):
        windowed_signal = self.apply_window_function(input_signal, self.windowFunction)
        spectrum = self.frequency_spectrum(windowed_signal)
        amplitudes = np.ones((len(self.bands), self.channelCount), dtype='float')
        # amplitudes = []
        # for wave, band_range in self.bands.items():
        for i, band in enumerate(self.bands):
            # band_amplitude = self.avg_band_amplitude(spectrum, int(band_range[0]), int(band_range[1]))
            band_amplitude = self.avg_band_amplitude(spectrum, int(self.bands[band][0]), int(self.bands[band][1]))
            if any(np.isnan(band_amplitude)):
                for index, channel_amplitude in enumerate(band_amplitude):
                    if np.isnan(channel_amplitude):
                        band_amplitude[index] = 0.
                amplitudes[i, :] = band_amplitude
                # amplitudes.append(band_amplitude.flatten())
                print("No frequencies exist in this bandwidth")
            else:
                amplitudes[i, :] = band_amplitude
                # amplitudes.append(band_amplitude.flatten())
        return amplitudes

    def process(self):

        for chunkIndex in range(len(self.input[0])):
            if (type(self.input[0][chunkIndex]) == OVSignalHeader):
                self.signalHeader = self.input[0].pop()

                outputHeader = OVSignalHeader(
                self.signalHeader.startTime,
                self.signalHeader.endTime,
                [len(self.bands), self.channelCount],
                self.bands.keys() + self.channelCount * [''],
                self.signalHeader.samplingRate)

                self.output[0].append(outputHeader)

            elif (type(self.input[0][chunkIndex]) == OVSignalBuffer):
                chunk = self.input[0].pop()
                numpyBuffer = np.array(chunk).reshape(tuple(self.signalHeader.dimensionSizes))
                self.signalBuffer = self.do(numpyBuffer).flatten().tolist()
                print(self.signalBuffer)
                chunk = OVSignalBuffer(chunk.startTime, chunk.endTime, self.signalBuffer)
                self.output[0].append(chunk)

            elif (type(self.input[0][chunkIndex]) == OVSignalEnd):
                self.output[0].append(self.input[0].pop())


box = AmplitudeExtractionBox()
