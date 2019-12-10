import random
import numpy as np
from numpy.random import *
from scipy.integrate import odeint


def Duffing_oscillator_ODE(x1range, x2range, numICs, tSpan, seed, type="z"):  # function X = PendulumFn(x1range, x2range, numICs, tSpan, seed, max_potential)
    # try some initial conditions for x1, x2
    np.random.seed(seed=seed)

    delta, beta, alpha = 0.5, -1, 1
    def dynsys(x, t):
        dydt = np.zeros_like(x)
        dydt[0] = x[1]  # x[1, :]
        dydt[1] = -delta * x[1] - x[0] * (beta + alpha * x[0] ** 2)
        # print(dydt)
        return dydt

    lenT = len(tSpan)  # 11, 500

    X = np.zeros((numICs * lenT, 2))

    # randomly start from x1range(1) to x1range(2)
    # x1 = (x1range[1] - x1range[0]) * rand() + x1range[0]

    # randomly start from x2range(1) to x2range(2)
    # x2 = (x2range[1] - x2range[0]) * rand() + x2range[0]
    # x1 = uniform(-2, 2)

    # randomly start from x2range(1) to x2range(2)
    # x2 = uniform(-2, 2)

    """ic = [x1, x2]
    temp = odeint(dynsys, ic, tSpan)
    X = temp"""
    if type == "y":
        lenT = len(tSpan) - 1
        count = 1
        for j in range(100 * numICs):  # j = 1:100*numICs
            x1 = uniform(x1range[0], x1range[1])
            x2 = uniform(x2range[0], x2range[1])
            ic = [x1, x2]
            temp = odeint(dynsys, ic, tSpan)
            # [T, temp] = odeint(dynsys, ic, tSpan)
            temp = temp[1:]

            X[(count - 1) * lenT: lenT + (count - 1) * lenT, :] = temp
            if count == numICs:
                break
            count = count + 1
        return X

    else:
        count = 1
        for j in range(100 * numICs):  # j = 1:100*numICs
            x1 = uniform(x1range[0], x1range[1])
            x2 = uniform(x2range[0], x2range[1])
            ic = [x1, x2]
            temp = odeint(dynsys, ic, tSpan)
            # [T, temp] = odeint(dynsys, ic, tSpan)

            X[(count - 1) * lenT: lenT + (count - 1) * lenT, :] = temp
            if count == numICs:
                break
            count = count + 1
        return X
