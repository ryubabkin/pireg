from pireg import PeriodicRegression
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt


def y_sin(X, A, w, f):
    return A * np.sin(X * w * 2 * pi + f)


x1 = np.linspace(1, 91, 9000)
x2 = np.linspace(1, 601, 60000)
y1 = y_sin(x1, 1, 6, 0) + y_sin(x1, 2, 1 / 6, pi / 21) + y_sin(x1, 3, 1 / 13, pi / 3 * 5.1) + y_sin(x1, 3, 3.1, pi / 5.1)
y2 = y_sin(x2, 1, 6, 0) + y_sin(x2, 2, 1 / 6, pi / 21) + y_sin(x2, 3, 1 / 13, pi / 3 * 5.1) + y_sin(x2, 3, 3.1, pi / 5.1)

reg = PeriodicRegression()
T,S,R = reg.fit(
    signal=y1,
    interval=x1,
    n_freq=4,
    q_freq=0.99,
    learning_rate=0.0001,
    n_iterations=500,
    keep=False,
    decompose=True,
    find_trend=True
)



reg.plot_spectrum(log=True)
reg.plot_loss()
pred = reg.predict(x2)
plt.figure()
plt.plot(x2, y2)
plt.plot(x2, pred, alpha=0.5)
plt.show()

plt.figure(figsize=(15,7))
plt.plot(y2)
plt.plot(S)
plt.plot(R)
plt.show()
