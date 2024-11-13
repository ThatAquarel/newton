import numpy as np
import matplotlib.pyplot as plt

points = np.array(
    [
        # [265, -10, -29, 1],
        # [-257, -1, -43, 1],
        # [9, 255, -28, 1],
        # [2, -276, -42, 1],
        # [0, 0, 222, 1],
        # [8, -8, -281, 1],
        [266.88867, -11.628906, -19.845703, 1],
        [-257.01758, -3.578125, -35.158203, 1],
        [8.890625, 258.03906, -25.59375, 1],
        [6.6933594, -273.93164, -32.021484, 1],
        [0.69140625, -2.2167969, 229.09766, 1],
        [7.4960938, -10.748047, -276.76953, 1],
    ],
    dtype=np.float32,
)

points = np.repeat(points, 2, axis=1)[:, 0:-1]
points[:, 1::2] **= 2

target = np.array(
    [
        [9.81, 0, 0, 1],
        [-9.81, 0, 0, 1],
        [0, 9.81, 0, 1],
        [0, -9.81, 0, 1],
        [0, 0, 9.81, 1],
        [0, 0, -9.81, 1],
    ],
    dtype=np.float32,
)

x, res, rank, s = np.linalg.lstsq(points, target)

print(x)
print(res)
print(rank)
print(s)

i_t = 1 / 256 * 9.81
e = np.mean(np.abs(np.diagonal(x[:-1:2]) - i_t) / i_t)

print(f"percent error: {e*100:.2f}%")

np.set_printoptions(formatter={"float": "{: 0.3f}".format})
print(points @ x)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

ax.scatter(0, 0, 0, marker="o", color="black")

t = points @ x
p = points[:, 0:-1:2]
ax.quiver(*p.T / p.max(), *t.T[:-1] / t.max(), color="blue")

plt.show()
