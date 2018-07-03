import numpy as np
import scipy.integrate as sp

T = 250
Q0 = 1
dt = 1
g = 9.8
isp = 346.8
R = 6378100
m0 = 505846
fuel = 474646
u0 = np.array([R, 0, 0, 0, m0])
a0 = 1
e0 = 1
M = 5.972 * 10 ** 24
G = 6.67408 * 10 ** (-11)


def f(force, time, y):
    i = int(time)
    print(i)
    dx = y[2]
    dy = y[3]
    dvx = force[2 * i] / y[4] - np.linalg.norm(force[2 * i:2 * i + 2]) / (y[4] * g * isp) - \
          R ** 2 * g * np.cos(np.arctan2(y[1], y[0])) / (np.linalg.norm(y[0:2]) ** 2)
    dvy = force[2 * i + 1] / y[4] - np.linalg.norm(force[2 * i:2 * i + 1]) / (y[4] * g * isp) - \
          R ** 2 * g * np.sin(np.arctan2(y[1], y[0])) / (np.linalg.norm(y[0:2]) ** 2)
    dm = np.linalg.norm(force[2 * i:2 * i + 1]) / (isp * g)
    return np.array([dx, dy, dvx, dvy, dm])


def values(force):
    def fun(time, y):
        return f(force, time, y)

    return sp.solve_ivp(fun, (0, T), u0,max_step=1, min_step=1).y


def q(force):
    theta = np.arctan2(values(force)[3], values(force)[2]) - np.arctan2(values(force)[1], values(force)[0])
    # theta is the angle between the position vector and the velocity vector, used to calculate the eccentricity e
    r = np.linalg.norm(values(force)[0:2])
    v = np.linalg.norm(values(force)[2, 4])
    e = np.sqrt(((r * v / G * M) - 1) ** 2 * np.sin(theta) ** 2 + np.cos(theta) ** 2)
    a = 1 / (2 / r - v ** 2 / (G * M))
    return (a - a0) ** 2 / a0 + (e - e0) ** 2 / e0
