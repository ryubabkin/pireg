import numpy as np
from numpy import pi, sin, cos
import core.core as _c


class Loss(object):
    def __init__(
            self,
            loss: str = 'mse',
            epsilon: float = 1.35
    ):
        self.loss = loss
        self.epsilon = epsilon

    "A * cos (2 * pi *f * fs * x - pi * sin(phi))"

    @staticmethod
    def dfs(x, f, fs, A, phi):
        return - A * f * 2 * pi * x * sin(f * 2 * pi * fs * x - phi)

    @staticmethod
    def dA(x, f, fs, phi):
        return cos(f * 2 * pi * fs * x - phi)

    @staticmethod
    def dphi(x, f, fs, A, phi):
        return A * pi * sin(f * 2 * pi * fs * x - phi)

    def grad_basis(self, x, f, W):
        return np.array([
            self.dfs(x, f, W[0], W[1], W[2]),
            self.dA(x, f, W[0], W[2]),
            self.dphi(x, f, W[0], W[1], W[2])
        ])

    def grad(self, err, x, f, W):
        grad = self.grad_basis(x, f, W)
        if self.loss == 'mse':
            return grad * err
        elif self.loss == 'mae':
            return grad * np.sign(err)
        elif self.loss == 'huber':
            if np.abs(err) <= self.epsilon:
                return grad * err
            else:
                return grad * 2 * self.epsilon * np.sign(err)
        elif self.loss == 'logcosh':
            return grad * np.tanh(err)


class Optimizer(object):
    def __init__(
            self,
            params: _c.DotDict,
    ):
        self.loss = params.loss
        self.n_iterations = params.n_iterations
        self.learning_rate = params.learning_rate
        self.decay = params.decay
        self.verbose = params.verbose
        self.info = "Optimizer class"
        self.epsilon = params.epsilon
        self.L = Loss(loss=self.loss, epsilon=self.epsilon)
        self.do_projection = params.use_constraints

    @staticmethod
    def error(X, Y, F, W):
        return _c.generate_signal(X, F, W) - Y

    def rmse(self, X, Y, F, W):
        return np.sqrt(np.mean(self.error(X, Y, F, W) ** 2))

    def mae(self, X, Y, F, W):
        return np.mean(np.abs(self.error(X, Y, F, W)))

    def update_cache(self, cache, grad) -> None:
        return None

    def run(self, X: np.ndarray, Y: np.ndarray, F: np.ndarray, W: np.ndarray) -> (None, None):
        return None, None

    @staticmethod
    def projection(W, N):
        W[np.arange(N) * 3 + 1] = np.abs(W[np.arange(N) * 3 + 1])
        W[np.arange(N) * 3 + 2] = np.abs(W[np.arange(N) * 3 + 2])
        W[np.arange(N) * 3 + 3] -= W[np.arange(N) * 3 + 3] // (2 * pi) * 2 * pi
        return W


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
        # IQR = 1.5 * (np.quantile(Y, 0.75) - np.quantile(Y, 0.25))
        error = []
        cache = np.zeros((3 * len(F) + 1, 1))
        for epoch in range(self.n_iterations):
            for i in range(len(X)):
                err = np.mean(self.error(X[i:i + 1], Y[i:i + 1], F, W))
                for n in range(len(F)):
                    grads = self.L.grad(err, X[i:i + 1], F[n], W[3 * n + 1:3 * n + 4])
                    cache[3 * n + 1:3 * n + 4] = self.update_cache(cache[3 * n + 1:3 * n + 4], grads)
                    W[3 * n + 1:3 * n + 4] -= self.learning_rate * cache[3 * n + 1:3 * n + 4]
                if self.do_projection:
                    W = self.projection(W, len(F))
            epoch_error = self.error(X, Y, F, W)
            if self.verbose:
                print(f"Epoch {epoch + 1}: MAE = {self.mae(X, Y, F, W)}, RMSE = {self.rmse(X, Y, F, W)}")
            error.append(epoch_error)
            # self.learning_rate = _c.tune_lr(self.learning_rate, epoch)

        return W, error


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
        error = []
        cache = np.zeros((3 * len(F) + 1, 1))
        for epoch in range(self.n_iterations):
            for i in range(len(X)):
                err = np.mean(self.error(X[i:i + 1], Y[i:i + 1], F, W))
                for n in range(len(F)):
                    grads = self.L.grad(err, X[i:i + 1], F[n], W[3 * n + 1:3 * n + 4])
                    cache[3 * n + 1:3 * n + 4] = self.update_cache(cache[3 * n + 1:3 * n + 4], grads)
                    W[3 * n + 1:3 * n + 4] -= self.learning_rate * grads / (np.sqrt(cache[3 * n + 1:3 * n + 4]) + 1e-9)
                if self.do_projection:
                    W = self.projection(W, len(F))
            epoch_error = self.error(X, Y, F, W)
            if self.verbose:
                print(f"Epoch {epoch + 1}: MAE = {self.mae(X, Y, F, W)}, RMSE = {self.rmse(X, Y, F, W)}")
            error.append(epoch_error)
            self.learning_rate = _c.tune_lr(self.learning_rate, epoch)
        return W, error


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
        error = []
        cache1, cache2 = np.zeros((3 * len(F) + 1, 1)), np.zeros((3 * len(F) + 1, 1))
        for epoch in range(self.n_iterations):
            for i in range(len(X)):
                err = np.mean(self.error(X[i:i + 1], Y[i:i + 1], F, W))
                for n in range(len(F)):
                    grads = self.L.grad(err, X[i:i + 1], F[n], W[3 * n + 1:3 * n + 4])
                    cache1[3 * n + 1:3 * n + 4] = self.update_cache1(cache1[3 * n + 1:3 * n + 4], grads)
                    cache2[3 * n + 1:3 * n + 4] = self.update_cache2(cache2[3 * n + 1:3 * n + 4], grads)
                    M1 = cache1[3 * n + 1:3 * n + 4] / (1 - self.d1 ** (epoch + 1))
                    M2 = cache2[3 * n + 1:3 * n + 4] / (1 - self.d2 ** (epoch + 1))
                    W[3 * n + 1:3 * n + 4] -= self.learning_rate * M1 / (np.sqrt(M2) + 1e-9)
                if self.do_projection:
                    W = self.projection(W, len(F))
            epoch_error = self.error(X, Y, F, W)
            if self.verbose:
                print(f"Epoch {epoch + 1}: MAE = {self.mae(X, Y, F, W)}, RMSE = {self.rmse(X, Y, F, W)}")
            error.append(epoch_error)
            self.learning_rate = _c.tune_lr(self.learning_rate, epoch)
        return W, error


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
        return np.maximum(self.d2 * cache, np.abs(grad))

    def run(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            F: np.ndarray,
            W: np.ndarray,
    ) -> (np.ndarray, list):
        error = []
        cache1, cache2 = np.zeros((3 * len(F) + 1, 1)), np.zeros((3 * len(F) + 1, 1))
        for epoch in range(self.n_iterations):
            for i in range(len(X)):
                err = np.mean(self.error(X[i:i + 1], Y[i:i + 1], F, W))
                for n in range(len(F)):
                    grads = self.L.grad(err, X[i:i + 1], F[n], W[3 * n + 1:3 * n + 4])
                    cache1[3 * n + 1:3 * n + 4] = self.update_cache1(cache1[3 * n + 1:3 * n + 4], grads)
                    cache2[3 * n + 1:3 * n + 4] = self.update_cache2(cache2[3 * n + 1:3 * n + 4], grads)
                    M1 = cache1[3 * n + 1:3 * n + 4] / (1 - self.d1 ** (epoch + 1))
                    M2 = cache2[3 * n + 1:3 * n + 4]
                    W[3 * n + 1:3 * n + 4] -= self.learning_rate * M1 / (M2 + 1e-9)
                if self.do_projection:
                    W = self.projection(W, len(F))
            epoch_error = self.error(X, Y, F, W)
            if self.verbose:
                print(f"Epoch {epoch + 1}: MAE = {self.mae(X, Y, F, W)}, RMSE = {self.rmse(X, Y, F, W)}")
            error.append(epoch_error)
            self.learning_rate = _c.tune_lr(self.learning_rate, epoch)
        return W, error


class MIXED(Optimizer):
    def __init__(
            self,
            params: _c.DotDict
    ):
        Optimizer.__init__(self, params)
        self.info = 'SGDM + PMSProp'
        if type(self.decay) != tuple:
            raise ValueError('Parameter "decay" must be a tuple for mixed optimizer')
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
        error = []
        indx = np.arange(len(X))
        cache1, cache2 = np.zeros((3 * len(F) + 1, 1)), np.zeros((3 * len(F) + 1, 1))
        for epoch in range(self.n_iterations):
            np.random.shuffle(indx)
            portion = 1 / 2
            indx1, indx2 = indx[:int(len(indx) * portion)], indx[int(len(indx) * portion):]
            for i in range(len(indx1)):
                err = np.mean(self.error(X[indx1][i:i + 1], Y[indx1][i:i + 1], F, W))
                for n in range(len(F)):
                    grads = self.L.grad(err, X[indx1][i:i + 1], F[n], W[3 * n + 1:3 * n + 4])
                    cache1[3 * n + 1:3 * n + 4] = self.update_cache1(cache1[3 * n + 1:3 * n + 4], grads)
                    W[3 * n + 1:3 * n + 4] -= self.learning_rate * cache1[3 * n + 1:3 * n + 4]
                if self.do_projection:
                    W = self.projection(W, len(F))
            for i in range(len(indx2)):
                err = np.mean(self.error(X[indx2][i:i + 1], Y[indx2][i:i + 1], F, W))
                for n in range(len(F)):
                    grads = self.L.grad(err, X[indx2][i:i + 1], F[n], W[3 * n + 1:3 * n + 4])
                    cache2[3 * n + 1:3 * n + 4] = self.update_cache2(cache2[3 * n + 1:3 * n + 4], grads)
                    W[3 * n + 1:3 * n + 4] -= self.learning_rate * grads / (np.sqrt(cache2[3 * n + 1:3 * n + 4]) + 1e-9)
                if self.do_projection:
                    W = self.projection(W, len(F))
            epoch_error = self.error(X, Y, F, W)
            if self.verbose:
                print(f"Epoch {epoch + 1}: MAE = {self.mae(X, Y, F, W)}, RMSE = {self.rmse(X, Y, F, W)}")
            error.append(epoch_error)
            self.learning_rate = _c.tune_lr(self.learning_rate, epoch)
        return W, error
