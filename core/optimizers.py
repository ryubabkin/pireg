import numpy as np
from numpy import pi, sin, cos
import core.core as _c


class Optimizer(object):
    def __init__(
            self,
            params: _c.DotDict,
    ):
        self.n_iterations = params.n_iterations
        self.learning_rate = params.learning_rate
        self.decay = params.decay
        self.verbose = params.verbose
        self.info = "Optimizer class"

    @staticmethod
    def dfs(x, f, fs, A, B):
        return A * f * 2 * pi * x * cos(f * 2 * pi * fs * x) - B * f * 2 * pi * x * sin(f * 2 * pi * fs * x)

    @staticmethod
    def dA(x, f, fs):
        return sin(f * 2 * pi * fs * x)

    @staticmethod
    def dB(x, f, fs):
        return cos(f * 2 * pi * fs * x)

    @staticmethod
    def error(X, Y, F, W):
        return _c.generate_signal(X, F, W) - Y

    def rmse(self, X, Y, F, W):
        return np.sqrt(np.mean(self.error(X, Y, F, W) ** 2))

    def calc_grad(self, err, x, f, W):
        return np.array([
            self.dfs(x, f, W[0], W[1], W[2]),
            self.dA(x, f, W[0]),
            self.dB(x, f, W[0])
        ]) * err

    def update_cache(self, cache, grad) -> None:
        return None

    def run(self, X: np.ndarray, Y: np.ndarray, F: np.ndarray, W: np.ndarray) -> (None, None):
        return None, None


class SGD(Optimizer):
    def __init__(
            self,
            params: _c.DotDict
    ):
        Optimizer.__init__(self, params)
        self.info = 'Stochastic Gradient Descent with Momentum'
        if type(self.decay) == tuple:
            self.decay = self.decay[0]
            print(f'(!) decay value was set to {self.decay} for sgd optimizer')

    def update_cache(self, cache, grad) -> np.ndarray:
        return self.decay * cache + (1 - self.decay) * grad

    def run(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        F: np.ndarray,
        W: np.ndarray,
    ) -> (np.ndarray, list):
        loss = []
        cache = np.zeros((3 * len(F) + 1, 1))
        for epoch in range(self.n_iterations):
            for i in range(len(X)):
                err = np.mean(self.error(X[i:i + 1], Y[i:i + 1], F, W))
                for n in range(len(F)):
                    grads = self.calc_grad(err, X[i:i + 1], F[n], W[3 * n + 1:3 * n + 4])
                    cache[3 * n + 1:3 * n + 4] = self.update_cache(cache[3 * n + 1:3 * n + 4], grads)
                    W[3 * n + 1:3 * n + 4] -= self.learning_rate * cache[3 * n + 1:3 * n + 4]
                W[0] -= self.learning_rate * err
            epoch_loss = self.rmse(X, Y, F, W)
            if self.verbose:
                print(f"Epoch {epoch+1}: loss = {epoch_loss}")
            loss.append(epoch_loss)
        return W, loss


class RMSProp(Optimizer):
    def __init__(
            self,
            params: _c.DotDict
    ):
        Optimizer.__init__(self, params)
        self.info = 'Root Mean Square Propagation'
        if type(self.decay) == tuple:
            self.decay = self.decay[0]
            print(f'(!) decay value was set to {self.decay} for rmsprop optimizer')

    def update_cache(self, cache, grad) -> np.ndarray:
        return self.decay * cache + (1 - self.decay) * grad ** 2

    def run(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            F: np.ndarray,
            W: np.ndarray,
    ) -> (np.ndarray, list):
        loss = []
        cache = np.zeros((3 * len(F) + 1, 1))
        for epoch in range(self.n_iterations):
            for i in range(len(X)):
                err = np.mean(self.error(X[i:i + 1], Y[i:i + 1], F, W))
                for n in range(len(F)):
                    grads = self.calc_grad(err, X[i:i + 1], F[n], W[3 * n + 1:3 * n + 4])
                    cache[3 * n + 1:3 * n + 4] = self.update_cache(cache[3 * n + 1:3 * n + 4], grads)
                    W[3 * n + 1:3 * n + 4] -= self.learning_rate * grads / (np.sqrt(cache[3 * n + 1:3 * n + 4]) + 1e-9)
                W[0] -= self.learning_rate * err
            epoch_loss = self.rmse(X, Y, F, W)
            if self.verbose:
                print(f"Epoch {epoch+1}: loss = {epoch_loss}")
            loss.append(epoch_loss)
        return W, loss


class Adam(Optimizer):
    def __init__(
            self,
            params: _c.DotDict
    ):
        Optimizer.__init__(self, params)
        self.info = 'Adaptive Movement Estimation'
        if type(self.decay) != tuple:
            raise ValueError('Parameter "decay" must be a tuple for adam optimizer')
        self.d1, self.d2 = self.decay[0], self.decay[1]

    def update_cache1(self, cache, grad) -> np.ndarray:
        return self.d1 * cache + (1 - self.d1) * grad

    def update_cache2(self, cache, grad) -> np.ndarray:
        return self.d2 * cache + (1 - self.d2) * grad ** 2

    def run(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            F: np.ndarray,
            W: np.ndarray,
    ) -> (np.ndarray, list):
        loss = []
        cache1, cache2 = np.zeros((3 * len(F) + 1, 1)), np.zeros((3 * len(F) + 1, 1))
        for epoch in range(self.n_iterations):
            for i in range(len(X)):
                err = np.mean(self.error(X[i:i + 1], Y[i:i + 1], F, W))
                for n in range(len(F)):
                    grads = self.calc_grad(err, X[i:i + 1], F[n], W[3 * n + 1:3 * n + 4])
                    cache1[3 * n + 1:3 * n + 4] = self.update_cache1(cache1[3 * n + 1:3 * n + 4], grads)
                    cache2[3 * n + 1:3 * n + 4] = self.update_cache2(cache2[3 * n + 1:3 * n + 4], grads)
                    M1 = cache1[3 * n + 1:3 * n + 4] / (1 - self.d1 ** (epoch + 1))
                    M2 = cache2[3 * n + 1:3 * n + 4] / (1 - self.d2 ** (epoch + 1))
                    W[3 * n + 1:3 * n + 4] -= self.learning_rate * M1 / (np.sqrt(M2) + 1e-9)
                W[0] -= self.learning_rate * err
            epoch_loss = self.rmse(X, Y, F, W)
            if self.verbose:
                print(f"Epoch {epoch+1}: loss = {epoch_loss}")
            loss.append(epoch_loss)
        return W, loss


class AdaMax(Optimizer):
    def __init__(
            self,
            params: _c.DotDict
    ):
        Optimizer.__init__(self, params)
        self.info = 'Adaptive Movement Estimation Based on the Infinity Norm'
        if type(self.decay) != tuple:
            raise ValueError('Parameter "decay" must be a tuple for adamax optimizer')
        self.d1, self.d2 = self.decay[0], self.decay[1]

    def update_cache1(self, cache, grad) -> np.ndarray:
        return self.d1 * cache + (1 - self.d1) * grad

    def update_cache2(self, cache, grad) -> np.ndarray:
        return np.maximum(self.d2 * cache,  np.abs(grad))

    def run(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            F: np.ndarray,
            W: np.ndarray,
    ) -> (np.ndarray, list):
        loss = []
        cache1, cache2 = np.zeros((3 * len(F) + 1, 1)), np.zeros((3 * len(F) + 1, 1))
        for epoch in range(self.n_iterations):
            for i in range(len(X)):
                err = np.mean(self.error(X[i:i + 1], Y[i:i + 1], F, W))
                for n in range(len(F)):
                    grads = self.calc_grad(err, X[i:i + 1], F[n], W[3 * n + 1:3 * n + 4])
                    cache1[3 * n + 1:3 * n + 4] = self.update_cache1(cache1[3 * n + 1:3 * n + 4], grads)
                    cache2[3 * n + 1:3 * n + 4] = self.update_cache2(cache2[3 * n + 1:3 * n + 4], grads)
                    M1 = cache1[3 * n + 1:3 * n + 4] / (1 - self.d1 ** (epoch + 1))
                    M2 = cache2[3 * n + 1:3 * n + 4]
                    W[3 * n + 1:3 * n + 4] -= self.learning_rate * M1 / (M2 + 1e-9)
                W[0] -= self.learning_rate * err
            epoch_loss = self.rmse(X, Y, F, W)
            if self.verbose:
                print(f"Epoch {epoch+1}: loss = {epoch_loss}")
            loss.append(epoch_loss)
        return W, loss