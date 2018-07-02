import numpy as np
import scipy.integrate as sp

F = np.array([])  # The values of the module and angle of the thrust vector over time, from t=0 to T
T = 10800
Q0 = 1
dt = 0.1
g = 9.8
isp = 1
R = 1
m0 = 1
u0 = np.array([R, 0, 0, 0, m0])
a0 = 1
e0 = 1
M = 1
G = 1


def module(y, i, k):
    return np.sqrt(y[i] ** 2 + y[k] ** 2)


def f(time, y):
    i = round(time / dt)
    dx = y[2]
    dy = y[3]
    dvx = F[2 * i] / y[4] - module(F, 2 * i, 2 * i + 1) / (y[4] * g * isp) - \
          R ** 2 * g * np.cos(np.arctan2(y[1], y[0])) / (module(y, 0, 1) ** 2)
    dvy = F[2 * i + 1] / y[4] - module(F, 2 * i, 2 * i + 1) / (y[4] * g * isp) - \
          R ** 2 * g * np.sin(np.arctan2(y[1], y[0])) / (module(y, 0, 1) ** 2)
    dm = module(y, 2 * i, 2 * i + 1) / (isp * g)
    return np.array([dx, dy, dvx, dvy, dm])


def values(t):
    return sp.solve_ivp(f, (0, t), u0, t_eval=[t]).y


def q(t):
    theta = np.arctan2(values(t)[3], values(t)[2]) - np.arctan2(values(t)[1], values(t)[0])
    # theta is the angle between the position vector and the velocity vector, used to calculate the eccentricity e
    r = module(values(t), 0, 1)
    v = module(values(t), 2, 3)
    e = np.sqrt(((r * v / G * M) - 1) ** 2 * np.sin(theta) ** 2 + np.cos(theta) ** 2)
    a = 1 / (2 / r - v ** 2 / (G * M))
    return (a - a0) ** 2 / a0 + (e - e0) ** 2 / e0
