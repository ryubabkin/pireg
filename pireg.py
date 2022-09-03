from typing import Union
import warnings
import pickle
import numpy as np

import core.optimizers as _o
import core.plots as _p
import core.core as _c
import core.validation as _v

warnings.filterwarnings("ignore")


class PeriodicRegression(object):
    def __init__(self):
        self._fitted = False
        self.verbose = True
        self.__module__ = "PeriodicRegressionClass"
        self.spectrum = None
        self.top_spectrum = None
        self.trend = None
        self.loss = []
        self.frequencies = None
        self.weights = None
        self.Fs = 1  # sampling frequency
        self.n_freq = 3
        self.q_freq = 0.95
        self.params = _c.DotDict()

    def fit(
            self,
            signal: np.ndarray,
            interval: np.ndarray = None,
            n_freq: int = 3,
            q_freq: float = 0.95,
            optimizer: str = 'sgd',
            learning_rate: float = 0.01,
            n_iterations: int = 100,
            decay: Union[float, tuple] = (0.9, 0.999),
            verbose: bool = True,
            drop: bool = False
    ):
        self.params = _c.DotDict({
            'optimizer': optimizer,
            'learning_rate': learning_rate,
            'n_iterations': int(n_iterations),
            'decay': decay,
            'verbose': verbose,
            'drop': drop
        })
        _v.check_input(
            length_Y=len(signal),
            length_X=len(interval),
            n_freq=n_freq,
            q_freq=q_freq,
            params=self.params
        )
        self.verbose = verbose
        if interval is None:
            interval = np.arange(0, len(signal))

        if not self._fitted or drop:
            self.Fs = len(interval) / np.max(interval)
            self.n_freq = int(n_freq)
            self.q_freq = q_freq
            self.spectrum, self.top_spectrum, self.frequencies = _c.calculate_spectrum(
                signal=signal,
                n_freq=self.n_freq,
                q_freq=self.q_freq
            )
            self._init_weights()

        self._optimize(
            Y=signal,
            X=interval
        )
        self._fitted = True
        return None

    def predict(
            self,
            interval: np.ndarray
    ) -> np.ndarray:
        signal = _c.generate_signal(
            X=interval,
            F=self.frequencies,
            W=self.weights
        )
        return signal

    def _init_weights(self):
        self.weights = np.array([self.Fs, 0, 0] * self.n_freq)

    def _optimize(
            self,
            X: np.ndarray,
            Y: np.ndarray
    ):
        if self.params.optimizer == 'sgd':
            self.weights, self.loss = _o.SGD(
                Y=Y,
                X=X,
                F=self.frequencies,
                W=self.weights,
                params=self.params
            )
        elif self.params.optimizer == 'rmsprop':
            self.weights, self.loss = _o.RMSProp(
                Y=Y,
                X=X,
                F=self.frequencies,
                W=self.weights,
                params=self.params
            )
        if self.params.optimizer == 'adam':
            self.weights, self.loss = _o.ADAM(
                Y=Y,
                X=X,
                F=self.frequencies,
                W=self.weights,
                params=self.params
            )
        else:
            print('Not implemented yet')

    def plot_loss(
            self,
            save_to: str = None,
            log: bool = False
    ):
        _p.plot_loss(
            loss=self.loss,
            save_to=save_to,
            log=log
        )

    def plot_spectrum(
            self,
            save_to: str = None,
            log: bool = False
    ):
        _p.plot_spectrum(
            spectrum=self.spectrum,
            top_spectrum=self.top_spectrum,
            save_to=save_to,
            log=log
        )

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
