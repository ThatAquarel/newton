import numpy as np
import matplotlib.pyplot as plt


data = np.load("1730691143_acc_export.npy")

plt.plot(range(len(data)), data[:, 1])
plt.plot(range(len(data)), data[:, 2])
plt.plot(range(len(data)), data[:, 3])


plt.plot(range(len(data)), data[:, 4])
plt.plot(range(len(data)), data[:, 5])
plt.plot(range(len(data)), data[:, 6])

plt.show()
