import numpy as np
from numpy import pi, sin, cos
import core.core as _c

"""
METRICS
"""


def error(X, Y, F, W):
    return _c.generate_signal(X, F, W) - Y


def rmse(X, Y, F, W):
    return np.sqrt(np.mean(error(X, Y, F, W) ** 2))


def mae(X, Y, F, W, i):
    return np.mean(error(X, Y, F, W))


"""
DERIVATIVES
"""


def dfs(x, f, fs, A, B):
    return A * f * 2 * pi * x * cos(f * 2 * pi * fs * x) - B * f * 2 * pi * x * sin(f * 2 * pi * fs * x)


def dA(x, f, fs):
    return sin(f * 2 * pi * fs * x)


def dB(x, f, fs):
    return cos(f * 2 * pi * fs * x)


"""
OPTIMIZERS
"""


def ADAM(
        X: np.ndarray,
        Y: np.ndarray,
        F: np.ndarray,
        W: np.ndarray,
        params: _c.DotDict,
        **kwargs
) -> (np.ndarray, list):
    n_iterations, learning_rate, decay = params.n_iterations, params.learning_rate, params.decay
    verbose = params.verbose
    if type(decay) != tuple:
        raise ValueError('Parameter "decay" must be a tuple for optimizer "adam"')
    d1, d2 = decay[0], decay[1]
    loss = []
    cache_v, cache_m = [0] + [0, 0, 0] * len(F), [0] + [0, 0, 0] * len(F)
    for epoch in range(n_iterations):
        for i in range(len(X)):
            err = error(X[i:i + 1], Y[i:i + 1], F, W)
            for n in range(len(F)):
                grad_fs = err * dfs(X[i:i + 1], F[n], W[3 * n + 1], W[3 * n + 2], W[3 * n + 3])
                grad_A = err * dA(X[i:i + 1], F[n], W[3 * n + 1])
                grad_B = err * dB(X[i:i + 1], F[n], W[3 * n + 1])

                cache_m[3 * n + 1] = d1 * cache_m[3 * n + 1] + (1 - d1) * grad_fs
                cache_m[3 * n + 1] = d1 * cache_m[3 * n + 2] + (1 - d1) * grad_A
                cache_m[3 * n + 3] = d1 * cache_m[3 * n + 3] + (1 - d1) * grad_B

                cache_v[3 * n + 1] = d2 * cache_v[3 * n + 1] + (1 - d2) * grad_fs ** 2
                cache_v[3 * n + 2] = d2 * cache_v[3 * n + 2] + (1 - d2) * grad_A ** 2
                cache_v[3 * n + 3] = d2 * cache_v[3 * n + 3] + (1 - d2) * grad_B ** 2

                m_fs = cache_m[3 * n + 1] / (1 - d1 ** (epoch + 1))
                m_A = cache_m[3 * n + 2] / (1 - d1 ** (epoch + 1))
                m_B = cache_m[3 * n + 3] / (1 - d1 ** (epoch + 1))

                v_fs = cache_v[3 * n + 1] / (1 - d2 ** (epoch + 1))
                v_A = cache_v[3 * n + 2] / (1 - d2 ** (epoch + 1))
                v_B = cache_v[3 * n + 3] / (1 - d2 ** (epoch + 1))

                W[3 * n + 1] -= np.mean(learning_rate * m_fs / (np.sqrt(v_fs) + 1e-9))
                W[3 * n + 2] -= np.mean(learning_rate * m_A / (np.sqrt(v_A) + 1e-9))
                W[3 * n + 3] -= np.mean(learning_rate * m_B / (np.sqrt(v_B) + 1e-9))
            W[0] -= np.mean(learning_rate * err)
        epoch_loss = rmse(X, Y, F, W)
        if verbose:
            print(f"Epoch {epoch}: loss = {epoch_loss}")
        loss.append(epoch_loss)
    return W, loss


def ADAMax(
        X: np.ndarray,
        Y: np.ndarray,
        F: np.ndarray,
        W: np.ndarray,
        params: _c.DotDict,
        **kwargs
) -> (np.ndarray, list):
    n_iterations, learning_rate, decay = params.n_iterations, params.learning_rate, params.decay
    verbose = params.verbose
    if type(decay) != tuple:
        raise ValueError('Parameter "decay" must be a tuple for optimizer "adamax"')
    d1, d2 = decay[0], decay[1]
    loss = []
    cache_v, cache_m = [0] + [0, 0, 0] * len(F), [0] + [0, 0, 0] * len(F)
    for epoch in range(n_iterations):
        for i in range(len(X)):
            err = error(X[i:i + 1], Y[i:i + 1], F, W)
            for n in range(len(F)):
                grad_fs = err * dfs(X[i:i + 1], F[n], W[3 * n + 1], W[3 * n + 2], W[3 * n + 3])
                grad_A = err * dA(X[i:i + 1], F[n], W[3 * n + 1])
                grad_B = err * dB(X[i:i + 1], F[n], W[3 * n + 1])

                cache_m[3 * n + 1] = d1 * cache_m[3 * n + 1] + (1 - d1) * grad_fs
                cache_m[3 * n + 2] = d1 * cache_m[3 * n + 2] + (1 - d1) * grad_A
                cache_m[3 * n + 3] = d1 * cache_m[3 * n + 3] + (1 - d1) * grad_B

                cache_v[3 * n + 1] = np.maximum(d2 * cache_v[3 * n + 1], np.abs(grad_fs))
                cache_v[3 * n + 2] = np.maximum(d2 * cache_v[3 * n + 2], np.abs(grad_A))
                cache_v[3 * n + 3] = np.maximum(d2 * cache_v[3 * n + 3], np.abs(grad_B))

                m_fs = cache_m[3 * n + 1] / (1 - d1 ** (epoch + 1))
                m_A = cache_m[3 * n + 2] / (1 - d1 ** (epoch + 1))
                m_B = cache_m[3 * n + 3] / (1 - d1 ** (epoch + 1))

                W[3 * n + 1] -= np.mean(learning_rate * m_fs / (cache_v[3 * n + 1] + 1e-9))
                W[3 * n + 2] -= np.mean(learning_rate * m_A / (cache_v[3 * n + 2] + 1e-9))
                W[3 * n + 3] -= np.mean(learning_rate * m_B / (cache_v[3 * n + 3] + 1e-9))
            W[0] -= np.mean(learning_rate * err)
        epoch_loss = rmse(X, Y, F, W)
        if verbose:
            print(f"Epoch {epoch}: loss = {epoch_loss}")
        loss.append(epoch_loss)
    return W, loss


def RMSProp(
        X: np.ndarray,
        Y: np.ndarray,
        F: np.ndarray,
        W: np.ndarray,
        params: _c.DotDict,
        **kwargs
) -> (np.ndarray, list):
    n_iterations, learning_rate, decay = params.n_iterations, params.learning_rate, params.decay
    verbose = params.verbose
    if type(decay) == tuple:
        decay = decay[0]
        print(f'decay value was set to {decay} for optimizer "rmsprop"')
    loss = []
    cache = [0] + [0, 0, 0] * len(F)
    for epoch in range(n_iterations):
        for i in range(len(X)):
            err = error(X[i:i + 1], Y[i:i + 1], F, W)
            for n in range(len(F)):
                grad_fs = err * dfs(X[i:i + 1], F[n], W[3 * n + 1], W[3 * n + 2], W[3 * n + 3])
                grad_A = err * dA(X[i:i + 1], F[n], W[3 * n + 1])
                grad_B = err * dB(X[i:i + 1], F[n], W[3 * n + 1])

                cache[3 * n + 1] = decay * cache[3 * n + 1] + (1 - decay) * grad_fs ** 2
                cache[3 * n + 2] = decay * cache[3 * n + 2] + (1 - decay) * grad_A ** 2
                cache[3 * n + 3] = decay * cache[3 * n + 3] + (1 - decay) * grad_B ** 2

                W[3 * n + 1] -= np.mean(learning_rate * grad_fs / (np.sqrt(cache[3 * n + 1] + 1e-9)))
                W[3 * n + 2] -= np.mean(learning_rate * grad_A / (np.sqrt(cache[3 * n + 2] + 1e-9)))
                W[3 * n + 3] -= np.mean(learning_rate * grad_B / (np.sqrt(cache[3 * n + 3] + 1e-9)))
            W[0] -= np.mean(learning_rate * err)
        epoch_loss = rmse(X, Y, F, W)
        if verbose:
            print(f"Epoch {epoch}: loss = {epoch_loss}")
        loss.append(epoch_loss)
    return W, loss


def SGD(
        X: np.ndarray,
        Y: np.ndarray,
        F: np.ndarray,
        W: np.ndarray,
        params: _c.DotDict,
        **kwargs
) -> (np.ndarray, list):
    n_iterations, learning_rate, decay = params.n_iterations, params.learning_rate, params.decay
    verbose = params.verbose
    if type(decay) == tuple:
        decay = decay[0]
        print(f'decay value was set to {decay} for optimizer "sgd"')
    loss = []
    cache = [0] + [0, 0, 0] * len(F)
    for epoch in range(n_iterations):
        for i in range(len(X)):
            err = error(X[i:i + 1], Y[i:i + 1], F, W)
            for n in range(len(F)):
                grad_fs = err * dfs(X[i:i + 1], F[n], W[3 * n + 1], W[3 * n + 2], W[3 * n + 3])
                grad_A = err * dA(X[i:i + 1], F[n], W[3 * n + 1])
                grad_B = err * dB(X[i:i + 1], F[n], W[3 * n + 1])

                cache[3 * n + 1] = decay * cache[3 * n + 1] + (1 - decay) * grad_fs
                cache[3 * n + 2] = decay * cache[3 * n + 2] + (1 - decay) * grad_A
                cache[3 * n + 3] = decay * cache[3 * n + 3] + (1 - decay) * grad_B

                W[3 * n + 1] -= np.mean(learning_rate * cache[3 * n + 1])
                W[3 * n + 2] -= np.mean(learning_rate * cache[3 * n + 2])
                W[3 * n + 3] -= np.mean(learning_rate * cache[3 * n + 3])
            W[0] -= np.mean(learning_rate * err)
        epoch_loss = rmse(X, Y, F, W)
        if verbose:
            print(f"Epoch {epoch}: loss = {epoch_loss}")
        loss.append(epoch_loss)
    return W, loss