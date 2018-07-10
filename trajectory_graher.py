import main
import matplotlib.pyplot as plt
import numpy as np
angles = []
dtheta = 3.14 / 100
theta = 0
F = np.zeros(int(main.T / main.dt) * 2 + 2)
modules = np.zeros(int(main.T / main.dt) + 1)
for i in range(0, int(main.T / main.dt)):
    if i<main.T/main.dt/6:
        F[2 * i], F[2 * i + 1] = np.cos(0), np.sin(0)
    else:
        F[2 * i], F[2 * i + 1] = np.cos(np.pi/2), np.sin(np.pi/2)
    modules[i] = np.linalg.norm([F[2 * i], F[2 * i + 1]])
F = F * main.g * main.isp * main.fuel / main.T
X = []
Y = []
for i in range(0,200):
    X.append(main.R * np.cos(theta))
    Y.append(main.R * np.sin(theta))
    theta += dtheta
u = main.values(F)
x = u[0]
y = u[1]
m = u[4]
t = np.linspace(0, main.T, np.size(m))
#plt.plot(t, m)
plt.plot(x, y)
plt.plot(X, Y)
plt.xlim(-10 * main.R, 10 * main.R)
plt.ylim(-10 * main.R, 10 * main.R)
plt.show()
print(main.q(main.final_values(F)))