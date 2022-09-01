import warnings
import pickle
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, sin, cos
from scipy.signal import find_peaks, peak_widths

warnings.filterwarnings("ignore")


class DotDict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """

    def __getattr__(self, attr):
        if attr.startswith('__'):
            raise AttributeError
        return self.get(attr, None)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _generate_signal(X, F, W):
    total = np.zeros(len(X))
    for n in range(len(F)):
        total += W[3 * n + 1] * sin(2 * pi * F[n] * W[3 * n] * X) + W[3 * n + 2] * cos(2 * pi * F[n] * W[3 * n] * X)
    return total


def _error(X, Y, F, W):
    return _generate_signal(X, F, W) - Y


def _rmse(X, Y, F, W):
    return np.sqrt(np.mean(_error(X, Y, F, W) ** 2))


def _mae(X, Y, F, W):
    return np.mean(_error(X, Y, F, W))


def _dfs(x, f, fs, A, B):
    return A * f * 2 * pi * x * cos(f * 2 * pi * fs * x) - B * f * 2 * pi * x * sin(f * 2 * pi * fs * x)


def _dA(x, f, fs):
    return sin(f * 2 * pi * fs * x)


def _dB(x, f, fs):
    return cos(f * 2 * pi * fs * x)


def _get_trend(
        signal: np.ndarray
) -> (np.ndarray, np.ndarray):
    polyvals = np.polyfit(
        x=np.arange(len(signal)),
        y=signal,
        deg=1
    )
    flat = np.poly1d(polyvals.flatten())
    trend = flat(np.arange(len(signal)))
    polyvals = polyvals.flatten().tolist()
    return trend, polyvals


def _restore_trend(
        array: np.ndarray,
        polyvals: np.ndarray
) -> np.ndarray:
    restored = array * polyvals[0] + polyvals[1]
    return restored


"""
=============================================================
"""


class PeriodicRegression(object):
    def __init__(self):
        self.verbose = True
        self.__module__ = "PeriodicRegressionClass"
        self.spectrum = None
        self.top_spectrum = None
        self.trend = None
        self.loss = []
        self.frequencies = None
        self.weights = None
        # spectrum params
        self.Fs = 1  # sampling frequency
        self.n_freq = 3  # top N frequencies to consider
        self.q_freq = 0.95  # quantile of peaks heights for peaks detection threshold
        # regression params
        self.learning_rate = 0.01  # learning rate for gradient descent
        self.n_iterations = 100  # number of iterations
        # trend polyvals
        self.polyvals = None

    def _check_init(
            self,
            length: int,  # signal length
            interval: np.ndarray,
            n_freq: int = 3,
            q_freq: float = 0.95,
    ):
        if length == 0:
            raise ValueError('Signal is empty')
        if n_freq <= 0:
            raise ValueError('"n_freq" value should be greater than zero.')
        if (q_freq <= 0) or (q_freq > 1):
            raise ValueError('"q_freq" value must be between 0 and 1.')
        if len(interval) != length:
            raise ValueError('Array lengths are incomparable')
        self.Fs = len(interval) / np.max(interval)
        self.n_freq = n_freq
        self.q_freq = q_freq

    def _calculate_spectrum(
            self,
            signal: np.ndarray
    ):
        frequency = np.fft.rfftfreq(signal.size)
        intensity = np.fft.rfft(signal, norm='ortho')
        amplitude = np.abs(intensity)
        phase = np.angle(intensity)
        min_height = np.quantile(amplitude, 0.95)
        peaks, _ = find_peaks(amplitude, height=min_height)
        is_peak = np.zeros_like(frequency)
        is_peak[peaks] = 1
        half_width = np.zeros_like(frequency)
        half_width[peaks] = peak_widths(amplitude, peaks, rel_height=0.5)[0]
        self.spectrum = np.array([frequency, amplitude, phase, is_peak, half_width])
        top_spectrum = self.spectrum[:, self.spectrum[3] == 1]
        self.top_spectrum = top_spectrum[:, np.argsort(top_spectrum[1, :])][:, -int(self.n_freq):]
        self.frequencies = self.top_spectrum[0, :]
        n = len(self.frequencies)
        if n < int(self.n_freq):
            if n > 1:
                print(f'(WARNING) Only {n} frequencies were found. n_freq is set to {n}')
            if n == 1:
                print(f'(WARNING) Only 1 frequency was found. n_freq is set to 1')
            if n == 0:
                raise ValueError('No frequencies found')
        print("Spectrum was calculated. Use:\n"
              "*.spectrum to get the whole spectrum\n"
              f"*.top_spectrum to get top {int(self.n_freq)} peaks spectrum\n"
              f"*.plot_spectrum() to view the result\n")

    def _gradient_descent(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            learning_rate: float = 0.01,
            n_iterations: int = 100,
    ):
        self.loss = []
        for epoch in range(n_iterations):
            for i in range(len(X)):
                err = _error(
                    X=X[i:i + 1],
                    Y=Y[i:i + 1],
                    F=self.frequencies,
                    W=self.weights
                )
                for n in range(len(self.frequencies)):
                    self.weights[3 * n + 0] -= np.mean(learning_rate * err * _dfs(x=X[i:i + 1],
                                                                                  f=self.frequencies[n],
                                                                                  fs=self.weights[3 * n],
                                                                                  A=self.weights[3 * n + 1],
                                                                                  B=self.weights[3 * n + 2]
                                                                                  )
                                                       )
                    self.weights[3 * n + 1] -= np.mean(learning_rate * err * _dA(x=X[i:i + 1],
                                                                                 f=self.frequencies[n],
                                                                                 fs=self.weights[3 * n],
                                                                                 )
                                                       )
                    self.weights[3 * n + 2] -= np.mean(learning_rate * err * _dB(x=X[i:i + 1],
                                                                                 f=self.frequencies[n],
                                                                                 fs=self.weights[3 * n],
                                                                                 )
                                                       )
            epoch_loss = _rmse(X, Y, self.frequencies, self.weights)
            if self.verbose:
                print(f"Epoch {epoch}: loss = {epoch_loss}")
            self.loss.append(epoch_loss)

    def _init_weights(self):
        self.weights = np.array([self.Fs, 0, 0] * self.n_freq)

    def _detrend(
            self,
            signal: np.ndarray
    ):
        trend, self.polyvals = _get_trend(signal=signal)
        signal = signal - trend
        return trend, signal

    def fit(
            self,
            signal: np.ndarray,
            interval: np.ndarray = None,
            n_freq: int = 3,
            q_freq: float = 0.95,
            learning_rate: float = 0.01,
            n_iterations: int = 100,
            keep: bool = False,
            find_trend: bool = False,
            decompose: bool = False,
            verbose: bool = True

    ):
        self.verbose = verbose
        if interval is None:
            interval = np.arange(0, len(signal))

        if find_trend:
            trend, signal = self._detrend(signal=signal)
        else:
            trend = np.zeros_like(signal)

        if (keep and (self.spectrum is None)) or not keep:
            self._check_init(
                length=len(signal),
                interval=interval,
                n_freq=n_freq,
                q_freq=q_freq
            )
            self._calculate_spectrum(
                signal=signal
            )
            self._init_weights()

        self._gradient_descent(
            Y=signal,
            X=interval,
            learning_rate=learning_rate,
            n_iterations=n_iterations
        )

        if decompose:
            restored = _generate_signal(
                X=interval,
                F=self.frequencies,
                W=self.weights
            )
            residuals = signal - trend - restored
            return trend, restored, residuals
        return None

    def predict(
            self,
            interval: np.ndarray
    ) -> np.ndarray:
        signal = _generate_signal(
            X=interval,
            F=self.frequencies,
            W=self.weights
        )
        if self.polyvals is not None:
            trend = _restore_trend(
                array=interval,
                polyvals=self.polyvals
            )
            return signal + trend
        else:
            return signal

    def plot_loss(
            self,
            save_to: str = None,
            log: bool = False
    ):
        plt.figure(figsize=(10, 7))
        plt.plot(self.loss, c='blue')
        plt.axhline(0, c='lightgray')
        if log:
            plt.yscale('log')
            plt.ylabel('RMSE loss [log scale]', fontsize=15)
        else:
            plt.ylabel('RMSE loss', fontsize=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('Epochs', fontsize=15)
        plt.title(f'Loss {np.round(self.loss[-1],5)} for epoch {int(len(self.loss))}', fontsize=15)
        plt.tight_layout()
        if save_to:
            plt.savefig(save_to)
        plt.show()

    def plot_spectrum(
            self,
            save_to: str = None,
            log: bool = False
    ):
        plt.figure(figsize=(10, 7))
        plt.plot(self.spectrum[0, :],
                 self.spectrum[1, :], c='gray')
        plt.plot(self.spectrum[0, self.spectrum[3] == 1],
                 self.spectrum[1, self.spectrum[3] == 1],
                 'x', c='b')
        plt.plot(self.top_spectrum[0, :],
                 self.top_spectrum[1, :],
                 'x', c='r', markersize=10)
        if log:
            plt.xscale('log')
            plt.xlabel('Frequency (log scale)', fontsize=15)
        else:
            plt.xlabel('Frequency', fontsize=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylabel('Intensity [arb.units]', fontsize=15)
        plt.title('Signal spectrum', fontsize=15)
        plt.tight_layout()
        if save_to:
            plt.savefig(save_to)
        plt.show()

    def save_model(
            self,
            model_file: str
    ):
        file = open(model_file, 'wb')
        pickle.dump(self, file, -1)
        file.close()

    @staticmethod
    def load_model(
            model_file: str
    ):
        file = open(str(model_file), 'rb')
        model = pickle.load(file)
        file.close()
        return model
