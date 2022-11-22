from pireg import PeriodicRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('./tests/tempC.csv')
data = data.set_index('time')
data.index = pd.to_datetime(data.index)
D = data['tempC']

# D = data['Seattle']
D = D.sort_index()
X = ((D.index - D.index.min()).total_seconds() / 3600).astype(int)
Y = D.values

# lmodel = Ridge()
#
# lmodel.fit(X.values.reshape(-1, 1),
#            Y.reshape(-1, 1))
#
# Y_lin = lmodel.predict(X.values.reshape(-1, 1))
# Y = Y.reshape(-1, 1) - Y_lin
Y = Y.flatten()
x_train, y_train = X[:30000].values, Y[:30000]

# %%
reg = PeriodicRegression()
reg.fit(
    signal=y_train,
    interval=x_train,
    n_freq=2,
    learning_rate=5e-5,
    n_iterations=100,
    momentum=0.09,
    loss='huber',
    batch_size=4
)
reg.plot_spectrum(log=True)
plt.figure()
reg.plot_loss()
plt.show()
print(f'RMSE: {np.sqrt(np.mean(np.abs(y_train - y_train.mean()) ** 2))}')
print(f'MAE: {np.mean(np.abs(y_train - y_train.mean()))}')
#
pred = reg.predict(X.values)

plt.figure()
plt.plot(X, D.values)
plt.plot(X, pred, alpha=0.5)
plt.show()

# %% %%
