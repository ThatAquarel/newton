import numpy as np

def rotation_matrix(a, b, y):
    y_t = np.array([
        [np.cos(a), np.sin(a), 0, 0],
        [-np.sin(a), np.cos(a), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    p_t = np.array([
        [np.cos(b), 0, np.sin(b), 0],
        [0, 1, 0, 0],
        [-np.sin(b), 0, np.cos(b), 0],
        [0,0,0,1]
    ])
    r_t = np.array([
        [1, 0, 0, 0],
        [0, np.cos(y), np.sin(y), 0],
        [0, -np.sin(y), np.cos(y), 0],
        [0, 0,0,1]
    ])
    
    return y_t @ p_t @ r_t


deg_90_z = rotation_matrix(0, 0, np.pi/4)

point = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32)
print(point)
print(point @ deg_90_z)

t_t = np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [1, 2, 3, 1],
])

print(point @ deg_90_z @ t_t)
