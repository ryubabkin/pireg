from pireg import PeriodicRegression
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt


def y_sin(X, A, w, f):
    return A * np.sin(X * w * 2 * pi + f)


x1 = np.linspace(1, 121, 12000)
x2 = np.linspace(1, 601, 60000)
y1 = 10+y_sin(x1, 1, 1/20, 0) + y_sin(x1, 2, 1 / 6, pi / 21) + y_sin(x1, 3, 1 / 13, pi / 3 * 5.1) + y_sin(x1, 3, 3.1,pi / 5.1)
y2 = 10+y_sin(x2, 1, 1/20, 0) + y_sin(x2, 2, 1 / 6, pi / 21) + y_sin(x2, 3, 1 / 13, pi / 3 * 5.1) + y_sin(x2, 3, 3.1,pi / 5.1)

reg = PeriodicRegression()
reg.fit(
    signal=y1,
    interval=x1,
    n_freq=4,
    q_freq=0.99,
    learning_rate=0.0001,
    n_iterations=1,
    decay=(0.9, 0.999),
    optimizer='sgd'
)
reg.plot_spectrum(log=True)
reg.plot_loss()
pred = reg.predict(x1)

plt.figure()
plt.plot(x1, y1)
plt.plot(x1, pred, alpha=0.5)
plt.show()
#
# [[ 5.00000000e-04  1.66666667e-03]
#  [ 1.10469475e+00  2.20917871e+00]
#  [-1.25472993e+00 -3.68711403e-01]
#  [ 1.00000000e+00  1.00000000e+00]
#  [ 1.00022746e+00  1.00148910e+00]]
#
# [[ 5.00000000e-04  1.66666667e-03]
#  [ 1.10469475e+00  2.20917871e+00]
#  [-1.25472993e+00 -3.68711403e-01]
#  [ 1.00000000e+00  1.00000000e+00]
#  [ 1.00022746e+00  1.00148722e+00]]