import main
import numpy as np
import scipy.integrate as sp


def du_matrices(t, force, dF):
    i = round(t / main.dt)
    Fx, Fy = force[2 * i], force[2 * i + 1]
    F = np.linalg.norm([Fx, Fy])
    Vx, Vy = main.values(F, ts=[t])[2], main.values(F, ts=[t])[3]
    V = np.linalg.norm([Vx, Vy])
    rx, ry = main.values(F, ts=[t])[0], main.values(F, ts=[t])[1]
    R, g, isp = main.R, main.g, main.isp

    r = np.linalg.norm([rx, ry])
    m = main.values(F, ts=[t])[4]
    a = np.array([[0, 0, 1, 0, 0], [0, 0, 0, 1, 0],
                  [R ** 2 * g / (r ** 3) * (3 * rx ** 2 / (r ** 2) - 1), 3 * R ** 2 * g * rx * ry / (r ** 3),
                   -F / (m * isp * g), 0, 1 / m ** 2 * (F * Vx / (isp * g) - Fx)],
                  [3 * R ** 2 * g * rx * ry / (r ** 3), R ** 2 * g / (r ** 3) * (3 * ry ** 2 / (r ** 2) - 1), 0,
                   -F / (m * isp * g), 1 / m ** 2 * (F * Vy / (isp * g) - Fy)], [0, 0, 0, 0, 0]])
    b = np.array([[0, 0], [0, 0], [1 / m * (1 - Fx / (isp * g * F)), -Fy / (m * isp * g * F)],
                  [-Fx / (m * isp * g * F), 1 / m * (1 - Fy / (isp * g * F))],
                  [-Fx / (isp * g * F), -Fy / (isp * g * F)]])
    b = b @ np.array([dF[2 * i], dF[2 * i + 1]])
    return [a, b]


def f(t, du, force, dF):
    i = round(t / main.dt)
    return du_matrices(t, force, dF)[0] @ du + du_matrices(t, force, dF) @ np.array([dF[2 * i], dF[2 * i + 1]])


def calculate_du(t, force, dF):
    def func(time, du):
        return f(time, du, force, dF)
    return np.transpose(sp.solve_ivp(func, (0, main.T), [0, 0, 0, 0, 0], t_eval=[t]).y)[0]

def calculate_dq(values: np.ndarray, dvalues: np.ndarray):
    a = main.G * main.M
    theta = np.arctan2(values[3], values[2]) - np.arctan2(values[1], values[0])
    r = np.linalg.norm(values[0:2])
    dr = (values[0] * dvalues[0] + values[1] * dvalues[1]) / r
    v = np.linalg.norm(values[2:4])
    dv = (values[2] * dvalues[2] + values[3] * dvalues[3]) / v
    e = np.sqrt(((r * v / a) - 1) ** 2 * np.sin(theta) ** 2 + np.cos(theta) ** 2)
    a = 1 / (2 / r - v ** 2 / a
    dtheta = 1/(r ** 2) * (values[1] * dvalues[0] - values[0] * dvalues[1]) + 1 / (v ** 2) * (values[2] * dvalues[3] - values[3] * dvalues[2])
    de = v * np.sin(theta) ** 2 * dr/ a + r * np.sin(theta) ** 2 * dv/a + 2*(r * v/a -2)*np.cos(theta) * np.sin(theta) * dtheta
    da = (-2 * dr/r ** 2 - 2 * v * dv/a) / (2/r - v ** 2 /a)
    return 2 * (a - main.a0) * da / main.a0 ** 2 + 2 * (e - main.e0) * de / main.e0 ** 2


u = np.array([1221, 5325, 10, 15, 5403])
du = np.array([1, 0.2, 0.02, 0.001, 12])

print(calculate_dq(u, du))