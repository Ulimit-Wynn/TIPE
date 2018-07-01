import numpy as np
import scipy.integrate as sp

F = np.array()  # The values of the module and angle of the thrust vector over time, from t=0 to T
t = 0
T = 10800
Q0 = 1
dt = 0.1
g = 9.8
isp = 1
R = 1
m0 = 1
u0 = np.array(R, 0, 0, 0, M)
a0 = 1
e0 = 1
M = 1
G = 1

def module(y, i, k):
    return np.sqrt(y[i] ** 2 + y[k] ** 2)


def f(t, y):
    i = round(t/dt)
    dx = y[2]
    dy = y[3]
    dvx = F[2*i] / y[5] - module(F, 2*i, 2*i+1) / (y[5] * g * isp) - \
          R ** 2 * g * np.cos(np.arctan2(y[1], y[0])) / (module(y, 0, 1) ** 2)
    dvy = y[2*i+1] / y[5] - module(F, 2*i, 2*i+1) / (y[5] * g * isp) - \
          R ** 2 * g * np.sin(np.arctan2(y[1], y[0])) / (module(y, 0, 1) ** 2)
    dm = module(y, 2*i, 2*i+1) / (isp * g)
    return np.array([dx, dy, dvx, dvy, dm])


def final_values():
    return np.array(sp.solve_ivp(f, T, u0))


def Q(F):
    theta = arctan2(final_values()[3], final_values()[2]) - np.arctan2(final_values()[1], final_values()[0])
    r = module(final_values(), 0, 1)
    v = module(final_values(), 2, 3)
    e = np.sqrt(((r*v/G*M)-1)**2 * np.sin(theta)**2 + np.cos(theta)**2
    a = 1/(2/r - v**2/(G*M))
    return (a-a0)**2/a0 + (e-e0)**2/e0