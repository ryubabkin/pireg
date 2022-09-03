import numpy as np
from numpy import pi, sin, cos
from scipy.signal import find_peaks, peak_widths


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


def generate_signal(
        X: np.ndarray,
        F: np.ndarray,
        W: np.ndarray
) -> np.ndarray:
    total = np.zeros(len(X))
    for n in range(len(F)):
        total += W[3 * n + 1] * sin(2 * pi * F[n] * W[3 * n] * X) + W[3 * n + 2] * cos(2 * pi * F[n] * W[3 * n] * X)
    return total


def calculate_spectrum(
        signal: np.ndarray,
        n_freq: int = 3,
        q_freq: float = 0.95
    ) -> (np.ndarray, np.ndarray, np.ndarray):
        frequency = np.fft.rfftfreq(signal.size)
        intensity = np.fft.rfft(signal, norm='ortho')
        amplitude = np.abs(intensity)
        phase = np.angle(intensity)
        min_height = np.quantile(amplitude, q_freq)
        peaks, _ = find_peaks(amplitude, height=min_height)
        is_peak = np.zeros_like(frequency)
        is_peak[peaks] = 1
        half_width = np.zeros_like(frequency)
        half_width[peaks] = peak_widths(amplitude, peaks, rel_height=0.5)[0]
        spectrum = np.array([frequency, amplitude, phase, is_peak, half_width])
        top_spectrum = spectrum[:, spectrum[3] == 1]
        top_spectrum = top_spectrum[:, np.argsort(top_spectrum[1, :])][:, -int(n_freq):]
        frequencies = top_spectrum[0, :]
        n = len(frequencies)
        if n < int(n_freq):
            if n > 1:
                print(f'(WARNING) Only {n} frequencies were found. n_freq is set to {n}')
            if n == 1:
                print(f'(WARNING) Only 1 frequency was found. n_freq is set to 1')
            if n == 0:
                raise ValueError('No frequencies found')
        print("Spectrum was calculated. Use:\n"
              "*.spectrum to get the whole spectrum\n"
              f"*.top_spectrum to get top {int(n_freq)} peaks spectrum\n"
              f"*.plot_spectrum() to view the result\n")
        return spectrum, top_spectrum, frequencies


def get_trend(
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


def restore_trend(
        array: np.ndarray,
        polyvals: np.ndarray
) -> np.ndarray:
    restored = array * polyvals[0] + polyvals[1]
    return restored

