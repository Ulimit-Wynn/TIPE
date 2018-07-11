import main
import differentiator as diff
import numpy as np
import scipy.optimize as optimize
import scipy.integrate as integrate


def cons(F, fuel):
    modules = np.zeros(np.size(F) // 2)
    for i in range(0, np.size(F) // 2):
        modules[i] = np.linalg.norm(F[2 * i: 2 * i + 2])
    return main.isp * main.g * fuel - integrate.trapz(modules, dx=main.dt)


def optimise(F0, fuel):
    return optimize.minimize(main.q, F0, jac=diff.grad_q, constraints={'type': 'ineq', 'fun': cons, 'args': (fuel,)})


print(optimise(main.F0, main.fuel))
