import core.core as _c


def check_input(
        length_Y: int,  # signal length
        length_X: int,  # interval length
        n_freq: int,
        q_freq: float,
        params: _c.DotDict
):
    if length_Y == 0:
        raise ValueError('Signal is empty')
    if n_freq <= 0:
        raise ValueError('"n_freq" value should be greater than zero')
    if (q_freq <= 0) or (q_freq > 1):
        raise ValueError('"q_freq" value must be between 0 and 1')
    if length_X != length_Y:
        raise ValueError('Array lengths are incomparable')
    if params.n_iterations <= 0:
        raise ValueError('"n_iterations" be greater than zero')
    if params.optimizer not in ['sgd', 'rmsprop', 'adam', 'adamax', 'mixed']:
        raise ValueError('"optimizer should be one of "sgd", "rmsprop", "adam" or "adamax"')
    if params.loss not in ['mse', 'huber', 'logcosh', 'mae']:
        raise ValueError('"optimizer should be one of "mse", "mae", "huber", or "logcosh"')

