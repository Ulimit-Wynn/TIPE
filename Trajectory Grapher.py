import Main
import matplotlib.pyplot as plt
import numpy as np
angles = []
dtheta = 3.14/Main.T/2
theta = 0
for i in range(0, Main.T + 1):
    if theta < 3.14/2:
        angles.append(theta + dtheta)
F = np.zeros(Main.T * 2 + 2)
modules = np.zeros(Main.T + 1)
for i in range(0, Main.T + 1):
    F[2 * i], F[2 * i + 1] = np.cos(angles[i]), np.sin(angles[i])
    modules[i] = np.linalg.norm([F[2 * i], F[2 * i + 1]])
F = F * Main.g * Main.isp * Main.fuel / Main.T


x = Main.values(F)[0]
y = Main.values(F)[1]
t = np.linspace(0, 250, 255)

plt.plot(x, y)
plt.show()
