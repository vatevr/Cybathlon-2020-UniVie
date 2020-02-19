import numpy as np
import scipy.linalg
import scipy.io
import scipy.signal


class PeakFrequency:

    def __init__(self, channels, samples, fs, bands=None):
        self.channels = channels
        self.samples = samples
        self.fs = fs
        self.dft = scipy.linalg.dft(samples)
        self.idft = np.linalg.inv(self.dft)
        self.hilbert = np.zeros(samples)
        if samples % 2 == 0:
            self.hilbert[0] = self.hilbert[samples // 2] = 1
            self.hilbert[1:samples // 2] = 2
        else:
            self.hilbert[0] = 1
            self.hilbert[1:(samples + 1) // 2] = 2
        if channels > 1:
            ind = [np.newaxis] * 2
            ind[-1] = slice(None)
            self.hilbert = self.hilbert[tuple(ind)]
        if bands is None:
            self.bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30), 'gamma': (30, 45)}
        else:
            self.bands = bands
        for band in self.bands:
            from_val = int(self.samples / self.fs * self.bands[band][0])
            to_val = int(self.samples / self.fs * self.bands[band][1])
            self.bands[band] = (from_val, to_val)

    def peak_method(self, signal, method='median'):
        inst_phase = np.unwrap(np.angle(signal))
        inst_freq = np.diff(inst_phase, axis=0) / (2 * np.pi) * self.fs
        if method is 'median':
            median = np.median(inst_freq, axis=0)
            return median
        if method is 'rms':
            rms = np.sqrt(np.median(inst_freq**2, axis=0))
            return rms
        return inst_freq

    def transform_dict(self, x, method='median'):
        x = np.asarray(x)
        x = x.T
        if x.shape[0] != self.samples and x.shape[1] != self.channels:
            raise ValueError("configs (", self.channels, ",", self.samples, ") do not match input dims ", x.shape)
        if np.iscomplexobj(x):
            raise ValueError("x is not a real signal.")
        H = self.dft.dot(x) * self.hilbert.T
        instant_frequency = dict()
        for band in self.bands:
            signal = np.zeros((self.samples, self.channels), dtype=complex)
            from_val = self.bands[band][0]
            to_val = self.bands[band][1]
            signal[from_val:to_val, :] = H[from_val:to_val, :]
            signal = signal.T.dot(self.idft).T
            instant_frequency[band] = self.peak_method(signal, method)
        return instant_frequency

    def transform(self, x, method='median'):
        x = np.asarray(x)
        x = x.T
        if x.shape[0] != self.samples and x.shape[1] != self.channels:
            raise ValueError("configs (", self.channels, ",", self.samples, ") do not match input dims ", x.shape)
        if np.iscomplexobj(x):
            raise ValueError("x is not a real signal.")
        H = self.dft.dot(x) * self.hilbert.T
        instant_frequency = []
        for band in self.bands:
            signal = np.zeros((self.samples, self.channels), dtype=complex)
            from_val = int(self.samples / self.fs * self.bands[band][0])
            to_val = int(self.samples / self.fs * self.bands[band][1])
            signal[from_val:to_val, :] = H[from_val:to_val, :]
            signal = signal.T.dot(self.idft).T
            instant_frequency.append(self.peak_method(signal, method))
        return np.array(instant_frequency)