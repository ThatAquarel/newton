import numpy as np
import matplotlib.pyplot as plt

data = np.load("./collection/1730745063_acc_export.npy")
# data = np.load("./collection/1730745632_acc_export.npy")
# data = np.load("./collection/1730745752_acc_export.npy")

dt, acc_l, vel_a = data[:, 0], data[:, [1,2,3]], data[:, [4,5,6]]
acc_t = np.cumsum(dt)

def integrate(dx, y):
    dydx = (y[:-1] + y[1:]) * dx[1:, np.newaxis] / 2
    return np.cumsum(dydx, axis=0)

vel_t = acc_t[1:]
vel_l = integrate(dt, acc_l)
dis_a = integrate(dt, vel_a)

dis_t = acc_t[2:]
dis_l = integrate(dt[1:], vel_l)

fig, ax = plt.subplots(3, 2)

def plot_xyz(_ax, t, xyz):
    for i, color in enumerate(["red", "green", "blue"]):
        _ax.plot(t, xyz[:, i], color=color)

plot_xyz(ax[0, 0], acc_t, acc_l)
plot_xyz(ax[1, 0], vel_t, vel_l)
plot_xyz(ax[2, 0], dis_t, dis_l)

plot_xyz(ax[1, 1], acc_t, vel_a)
plot_xyz(ax[2, 1], vel_t, dis_a)

plt.show()


for rx, ry, rz in dis_a:    
    y, b, a = dis[3:6]
    y_t = np.array([
        [np.cos(a), np.sin(a), 0, 0],
        [-np.sin(a), np.cos(a), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    p_t = np.array([
        [np.cos(b), 0, -np.sin(b), 0],
        [0, 1, 0, 0],
        [np.sin(b), 0, np.cos(b), 0],
        [0,0,0,1]
    ])
    r_t = np.array([
        [1, 0, 0, 0],
        [0, np.cos(y), np.sin(y), 0],
        [0, -np.sin(y), np.cos(y), 0],
        [0, 0,0,1]
    ])
    t_t = np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [tx,ty,tz,1],
    ])
