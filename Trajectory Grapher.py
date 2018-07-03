import Main
import matplotlib.pyplot as plt
import numpy as np
angles = []
dtheta = 3.14/Main.T/2
theta = 0
F = np.zeros(int(Main.T/Main.dt) * 2 + 2)
modules = np.zeros(int(Main.T/Main.dt) + 1)
for i in range(0, int(Main.T/Main.dt)):
    if i<Main.T/Main.dt/2:
        F[2 * i], F[2 * i + 1] = np.cos(0), np.sin(0)
    else:
        F[2 * i], F[2 * i + 1] = np.cos(np.pi/2), np.sin(np.pi/2)
    modules[i] = np.linalg.norm([F[2 * i], F[2 * i + 1]])
F = F * Main.g * Main.isp * Main.fuel / Main.T


x = Main.values(F)[0]
y = Main.values(F)[1]

plt.plot(x, y)
plt.show()
