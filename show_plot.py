import numpy as np
import matplotlib.pyplot as plt


data = np.load("./collection/1730745632_acc_export.npy")
DATA_T = np.eye(7) * np.array([
    1e-6,
    *[1 / -256 * 9.81,] * 3,
    *[1 / 14.375 * np.pi / 180,] * 3
])

data = data @ DATA_T

fig, ax = plt.subplots(2, 1)

dt = data[:, 0]
accel_linear = data[:, [1,2,3]]
accel_angular = data[:, [4,5,6]]

t = np.cumsum(dt[::-1]) / 1e-6

for i, color in zip(range(3), ["red", "green", "blue"]):
    ax[0].plot(t, accel_linear[:, i], color=color)

for i, color in zip(range(3), ["red", "green", "blue"]):
    ax[1].plot(t, accel_angular[:, i], color=color)

plt.show()
