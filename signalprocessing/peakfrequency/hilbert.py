import numpy as np
from scipy.fftpack import ifft
import scipy.linalg
import scipy.io
import scipy.signal
from matplotlib import pyplot as plt


def dft_matrix(N):
    return scipy.linalg.dft(N)


def dft_matrix_2d(N):
    x = scipy.linalg.dft(N)
    return np.kron(x, x)


def hilbert_rotation(x, axis=-1):
    N = x.shape[axis]
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    if x.ndim > 1:
        ind = [np.newaxis] * x.ndim
        ind[axis] = slice(None)
        h = h[tuple(ind)]
    transformed = x * h
    return transformed

def find_peak(data):
    idx = np.unravel_index(np.argmax(data, axis=None), data.shape)
    return idx, data[idx]

def peak_frequency(x,  channels=None, dftmatrix=None, idftmatrix=None):
    N = x.shape[0]
    x = np.asarray(x)
    if channels is None:
        channels = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30), 'gamma': (30, 45)}
    if np.iscomplexobj(x):
        raise ValueError("x is not a real signal.")
    if dftmatrix is None:
        dftmatrix = dft_matrix(N)
        idftmatrix = np.linalg.inv(dftmatrix)
    xdft = dftmatrix.dot(x)
    H = hilbert_rotation(xdft)
    Z = dict()
    for channel in channels:
        signal = np.zeros(N, dtype=complex)
        signal[channels[channel][0]:channels[channel][1]] = H[channels[channel][0]:channels[channel][1]]

        Z[channel] = signal.dot(idftmatrix)

        plt.plot(np.imag(Z[channel]), label="imag")
        plt.plot(np.real(Z[channel]), label="real")
        plt.plot(np.abs(Z[channel]),label="absolute")
        plt.legend()
        plt.show()
    return Z


if __name__ == "__main__":
    MAT = scipy.io.loadmat('motor-imagery-eeg.mat')
    dict_keys = [*MAT.keys()]
    X = MAT[dict_keys[3]]
    data = X[0, :, 0]
    fs = 500
    plt.plot(data[:1000], label="data")
    plt.legend()
    plt.show()
    #peak_frequency(data[:1000],{'all': (1, 45)})
    analytic_signal = scipy.signal.hilbert(data)
    amplitude_envelope = np.abs(analytic_signal)
    plt.plot(data[:100],label="data")
    plt.plot(amplitude_envelope[:100],label="envelope")
    plt.legend()
    plt.show()
