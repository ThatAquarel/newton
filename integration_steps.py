import numpy as np
import matplotlib.pyplot as plt

# data = np.load("./collection/1730745063_acc_export.npy")
# data = np.load("./collection/1730745632_acc_export.npy")
data = np.load("./collection/1730745752_acc_export.npy")

dt, acc_l, vel_a = data[:, 0], data[:, [1, 2, 3]], data[:, [4, 5, 6]]
acc_t = np.cumsum(dt)


def integrate(dx, y):
    dydx = (y[:-1] + y[1:]) * dx[1:, np.newaxis] / 2
    return np.cumsum(dydx, axis=0)


vel_t = acc_t[1:]
vel_l = integrate(dt, acc_l)
dis_a = integrate(dt, vel_a)

dis_t = acc_t[2:]
dis_l = integrate(dt[1:], vel_l)

fig, ax = plt.subplots(3, 3)


def plot_xyz(_ax, t, xyz):
    for i, color in enumerate(["red", "green", "blue"]):
        _ax.plot(t, xyz[:, i], color=color)


plot_xyz(ax[0, 0], acc_t, acc_l)
plot_xyz(ax[1, 0], vel_t, vel_l)
plot_xyz(ax[2, 0], dis_t, dis_l)

plot_xyz(ax[1, 1], acc_t, vel_a)
plot_xyz(ax[2, 1], vel_t, dis_a)


def rotation_matrix(rz, ry, rx):
    y_t = np.array(
        [
            [np.cos(rz), np.sin(rz), 0, 0],
            [-np.sin(rz), np.cos(rz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    p_t = np.array(
        [
            [np.cos(ry), 0, np.sin(ry), 0],
            [0, 1, 0, 0],
            [-np.sin(ry), 0, np.cos(ry), 0],
            [0, 0, 0, 1],
        ]
    )
    r_t = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(rx), np.sin(rx), 0],
            [0, -np.sin(rx), np.cos(rx), 0],
            [0, 0, 0, 1],
        ]
    )

    t_t = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 9.81, 1],
        ]
    )

    return y_t @ p_t @ r_t @ t_t


acc_l, dis_a = acc_l[1:], dis_a[:]
# dis_l_T_dt = np.empty(dis_l.shape, dtype=dis_l.dtype)
acc_l_T_dt = np.empty(acc_l.shape, dtype=acc_l.dtype)

for i, (xyz_l, xyz_a) in enumerate(zip(acc_l, dis_a)):
    mat = rotation_matrix(*xyz_a[::-1])
    mat = np.linalg.inv(mat)
    xyz_l_T = xyz_l @ mat[0:3, 0:3]

    acc_l_T_dt[i, :] = xyz_l_T

# dis_l_T = np.cumsum(dis_l_T_dt, axis=0)
acc_l_T = acc_l_T_dt
dis_l_T_t = vel_t

plot_xyz(ax[0, 2], dis_l_T_t, acc_l_T)

plt.show()
