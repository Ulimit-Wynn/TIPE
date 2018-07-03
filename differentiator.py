import Main
import numpy as np


def du_matrices(t, force, dF):
    i = round(t / Main.dt)
    Fx, Fy = force[2 * i], force[2 * i + 1]
    F = np.linalg.norm([Fx, Fy])
    Vx, Vy = Main.values(F, ts=[t])[2], Main.values(F, ts=[t])[3]
    V = np.linalg.norm([Vx, Vy])
    rx, ry = Main.values(F, ts=[t])[0], Main.values(F, ts=[t])[1]
    R, g, isp = Main.R, Main.g, Main.isp

    r = np.linalg.norm([rx, ry])
    m = Main.values(F, ts=[t])[4]
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
