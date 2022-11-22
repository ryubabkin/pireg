import numpy as np
from numpy import pi, sin, cos
import core.core as _c
import matplotlib.pyplot as plt
import pandas as pd


class Loss(object):
    """
    {A} * cos (2 * pi * f * {cc} * X + {phi})
    """

    def __init__(
            self,
            loss: str = 'mse',
            epsilon: float = 1.35
    ):
        self.loss = loss
        self.epsilon = epsilon

    @staticmethod
    def dcc(x, f, cc, A, phi):
        return - A * f * 2 * pi * x * sin(f * 2 * pi * cc * x + phi)

    @staticmethod
    def dA(x, f, cc, phi):
        return cos(f * 2 * pi * cc * x + phi)

    @staticmethod
    def dphi(x, f, cc, A, phi):
        return - A * sin(f * 2 * pi * cc * x + phi)

    def grad_basis(self, x, f, W):
        return np.array([
            self.dcc(x, f, W[0], W[1], W[2]),
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
        self.A_init = None
        self.Cc = params.Cc
        self.batch_size = params.batch_size
        self.n_iterations = params.n_iterations
        self.learning_rate = params.learning_rate
        self.momentum = params.momentum
        self.verbose = params.verbose
        self.info = 'Stochastic Gradient Descent with Momentum'
        self.L = Loss(loss=params.loss, epsilon=params.epsilon)
        self.lr_decay = params.lr_decay
        self.c_max = None
        self.n_frequencies = None
        self.n = None

    @staticmethod
    def error(X, Y, F, W):
        return _c.generate_signal(X, F, W) - Y

    def rmse(self, X, Y, F, W):
        return np.sqrt(np.mean(self.error(X, Y, F, W) ** 2))

    def mae(self, X, Y, F, W):
        return np.mean(np.abs(self.error(X, Y, F, W)))

    @staticmethod
    def update_cache(mu, cache, grad) -> np.ndarray:
        return mu * cache + (1 - mu) * grad

    def set_params(self, X, F, W):
        self.n_frequencies = len(F)
        self.n = np.arange(self.n_frequencies) * 3

        # initial intensities
        self.A_init = W[np.arange(self.n_frequencies) * 3 + 1].copy()

        # projection params:
        d = max(X)-min(X)
        P = 0.99 - 4.9 / (d * F.reshape(-1, 1)) ** 1.5
        self.c_max = (1 / P) ** (1 / 1.5) * 1.01

    def run(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            F: np.ndarray,
            W: np.ndarray
    ) -> (np.ndarray, list):
        self.set_params(X, F, W)
        indxs = np.arange(len(X))
        RES = W.copy()
        error = []
        cache = np.zeros((3 * self.n_frequencies + 1, 1))
        for epoch in range(self.n_iterations):
            np.random.shuffle(indxs)
            lr = self.update_lr(epoch)
            mu = self.update_mu(epoch)
            for i in range(0, len(X), self.batch_size):
                Xs, Ys = X[indxs][i:i + self.batch_size], Y[indxs][i:i + self.batch_size]
                err = np.mean(self.error(Xs, Ys, F, W))
                for n in range(self.n_frequencies):
                    grads = np.mean(self.L.grad(err, Xs, F[n], W[3 * n + 1:3 * n + 4]), axis=1).reshape(-1, 1)
                    cache[3 * n + 1:3 * n + 4] = self.update_cache(mu, cache[3 * n + 1:3 * n + 4], grads)
                    W[3 * n + 1:3 * n + 4] -= lr * cache[3 * n + 1:3 * n + 4]
                W[0] -= lr * err
                W = self.projection(W)
                if i % 100 == 0:
                    RES = np.hstack((RES, W))
            epoch_error = self.error(X, Y, F, W)
            error.append(epoch_error)
            if self.verbose:
                # if epoch % 20 == 0:
                #     self.plot(X, Y, F, W, epoch)
                print(f"Epoch {epoch + 1}: "
                      f"MAE = {np.round(self.mae(X, Y, F, W),5)}, "
                      f"LR = {'{:.3e}'.format(lr)}, "
                      f"MU={'{:.3e}'.format(mu)}"
                      )
        self.plot(X, Y, F, W, epoch)
        pd.DataFrame(RES.T).to_csv('./res/R.csv', index=False)
        return W, error

    def projection(self, W):
        # Conversion coefficient (Cc) constraints

        W[self.n + 1] = np.where(W[self.n + 1] < self.Cc * 1, self.Cc * 1, W[self.n + 1])
        W[self.n + 1] = np.where(W[self.n + 1] > self.Cc * self.c_max, self.Cc * self.c_max, W[self.n + 1])

        # Intensity (A) constraints
        W[self.n + 2] = np.where(W[self.n + 2] < self.A_init, self.A_init, W[self.n + 2])

        # Phase (phi) constraints
        W[self.n + 3] -= W[self.n + 3] // (2 * pi) * 2 * pi

        return W

    def update_lr(self, i):
        p_t = 1 - i / self.n_iterations
        return self.learning_rate * p_t / ((1 - self.learning_rate) + self.learning_rate * p_t)

    def update_mu(self, i):
        p_t = 1 - i / self.n_iterations
        return self.momentum * p_t / ((1 - self.momentum) + self.momentum * p_t)

    @staticmethod
    def plot(X, Y, F, W, epoch):
        fig = plt.figure(figsize=(10, 7))
        pred = _c.generate_signal(X, F, W)
        plt.plot(X, Y)
        plt.plot(X, pred, alpha=0.5)
        ax = fig.add_axes([0.05, 0.05, 0.5, 0.3])
        ax.plot(X[1000:1300], Y[1000:1300])
        ax.plot(X[1000:1300], pred[1000:1300], alpha=0.5)
        plt.tight_layout()
        plt.savefig(f'./tests/plot/{epoch}.png')
        plt.show()
