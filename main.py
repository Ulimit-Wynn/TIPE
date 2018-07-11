import numpy as np
import scipy.integrate as sp

T = 1080
Q0 = 1
dt = 1
g = 9.8
isp = 346.8
R = 6378100
m0 = 546054
fuel = 514646
u0 = np.array([R, 0, 0, 0, m0])
a0 = 25271000
e0 = 0.66847
M = 5.972 * 10 ** 24
G = 6.67408 * 10 ** (-11)

F0 = np.zeros(2 * (T // dt + 1))
for i in range(T // dt + 1):
    if i <= T // dt / 6:
        F0[i] = np.cos(0)
        F0[i + 1] = np.sin(0)
    else:
        F0[i] = np.cos(np.pi / 2)
        F0[i + 1] = np.sin(np.pi / 2)


def normalised(F, norm):
    modules = np.zeros(np.size(F) // 2)
    for i in range(0, np.size(F) // 2):
        modules[i] = np.linalg.norm(F[2 * i: 2 * i + 2])
    return norm / sp.trapz(F, dx=dt) * F


F0 = normalised(F0, fuel * isp * g)


def density(r):
    if r <= R + 11000:
        return 1.225 * np.exp((-g * M * (r - R)) / (8.314 * 288.14))
    elif r <= R + 20000:
        return 0.363 * np.exp((-g * M * (r - R - 11000)) / (8.314 * 216.65))
    elif r <= R + 32000:
        return 0.088 * np.exp((-g * M * (r - R)) / (8.314 * 216.65))
    elif r <= R + 47000:
        return 0.013 * np.exp((-g * M * (r - R)) / (8.314 * 228.65))
    elif r <= R + 51000:
        return 0.00143 * np.exp((-g * M * (r - R)) / (8.314 * 270.65))
    elif r <= R + 71000:
        return 0.00086 * np.exp((-g * M * (r - R)) / (8.314 * 270.65))
    else:
        return 0.000064 * np.exp((-g * M * (r - R)) / (8.314 * 214.64))


def f(force, time, y):
    i = int(time / dt)
    dx = y[2]
    dy = y[3]
    dvx = force[2 * i] / y[4] - np.linalg.norm(np.array([force[2 * i], force[2 * i + 1]])) / (y[4] * g * isp) - \
          R ** 2 * g * np.cos(np.arctan2(y[1], y[0])) / (np.linalg.norm(y[0:2]) ** 3)
    dvy = force[2 * i + 1] / y[4] - np.linalg.norm(np.array([force[2 * i], force[2 * i + 1]])) / (y[4] * g * isp) - \
          R ** 2 * g * np.sin(np.arctan2(y[1], y[0])) / (np.linalg.norm(y[0:2]) ** 3)
    dm = -np.linalg.norm(np.array([force[2 * i], force[2 * i + 1]])) / (isp * g)
    return np.array([dx, dy, dvx, dvy, dm])


def values(force, ts=None):
    def fun(time, y):
        return f(force, time, y)

    return np.transpose(sp.solve_ivp(fun, (0, T), u0, t_eval=ts, max_step=10, min_step=10).y)[0]


def final_values(force):
    def fun(time, y):
        return f(force, time, y)

    return np.transpose(sp.solve_ivp(fun, (0, T), u0, t_eval=[T], max_step=10, min_step=10).y)[0]


def q(F):
    u = final_values(F)
    theta = np.arctan2(u[3], u[2]) - np.arctan2(u[1], u[0])
    # theta is the angle between the position vector and the velocity vector, used to calculate the eccentricity e
    r = np.linalg.norm(u[0:2])
    v = np.linalg.norm(u[2:4])
    e = np.sqrt(((r * v / G * M) - 1) ** 2 * np.sin(theta) ** 2 + np.cos(theta) ** 2)
    a = 1 / (2 / r - v ** 2 / (G * M))
    return (a - a0) ** 2 / (a0 ** 2) + (e - e0) ** 2 / (e0 ** 2)
