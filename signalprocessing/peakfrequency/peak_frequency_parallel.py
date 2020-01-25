import numpy as np
import scipy.linalg
import scipy.io
import scipy.signal
import threading


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
        self.instant_frequency = dict()

    def calculate(self, band, H):
        signal = np.zeros((self.samples, self.channels), dtype=complex)
        from_val = int(self.samples / self.fs * self.bands[band][0])
        to_val = int(self.samples / self.fs * self.bands[band][1])
        signal[from_val:to_val, :] = H[from_val:to_val, :]
        signal = signal.T.dot(self.idft).T
        inst_phase = np.unwrap(np.angle(signal))
        inst_freq = np.diff(inst_phase, axis=0) / (2 * np.pi) * self.fs
        self.instant_frequency[band] = np.median(inst_freq, axis=0)



    def fit(self, x):
        self.instant_frequency.clear()
        x = np.asarray(x)
        if x.shape[0] != self.samples and x.shape[1] != self.channels:
            raise ValueError("configs (", self.channels, ",", self.sample, ") do not match input dims ", x.shape)
        if np.iscomplexobj(x):
            raise ValueError("x is not a real signal.")
        H = self.dft.dot(x) * self.hilbert.T
        threads = []
        for band in self.bands:
            t = threading.Thread(
                target=self.calculate,
                args=(band, H))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        # returns a dict for each frequency band containing a channel vector for each electrode's median inst freq
        return self.instant_frequency