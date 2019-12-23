import random
import numpy as np
from numpy.random import *
from scipy.integrate import odeint


def Discrete_Linear_ODE(x1range, x2range, numICs, tSpan, seed, type="z"):
    # try some initial conditions for x1, x2
    np.random.seed(seed=seed)

    """a, b, c, d = 0.85, -0.05, -0.05, 0.85 # 1, -1, 0.02, 0.7  # mu = 0.8, 0.9
    def dynsys(x):
        dydt = np.zeros_like(x)
        dydt[0] = a * x[0] + b * x[1]
        dydt[1] = c * x[0] + d * x[1]
        return dydt"""

    def dynsys(x):
        dydt = np.zeros_like(x)
        dydt[0] = 0.9 * x[0] - 0.1 * x[1]
        dydt[1] = 0.8 * x[1]
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
            x1, x2 = np.random.normal(
                loc=0,  # 平均
                scale=1,  # 標準偏差
                size=2,  # 出力配列のサイズ(タプルも可)
            )
            ic = np.array([x1, x2])
            temp = ic.reshape(1, 2)
            for dis in range(1, len(tSpan)):
                tt = dynsys(temp[-1, :])
                temp = np.vstack((temp, tt))
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
            ic = np.array([x1, x2])
            temp = ic.reshape(1, 2)
            for dis in range(1, len(tSpan)):
                tt = dynsys(temp[-1, :])
                temp = np.vstack((temp, tt))
            # [T, temp] = odeint(dynsys, ic, tSpan)

            X[(count - 1) * lenT: lenT + (count - 1) * lenT, :] = temp
            if count == numICs:
                break
            count = count + 1
        return X
