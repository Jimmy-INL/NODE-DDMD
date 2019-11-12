import numpy as np
from scipy import linalg as la

A12 = np.array([[-1, 1, 0],
                [0, -1, 1],
                [1, 0, -1]])  # Unknown head nodes incidence matrix
A10 = 3



# Extended Dynamic Mode Decomposition
# Testing the concept
data = np.array([[1, 2],
                 [2, 5],
                 [3, 10],
                 [4, 17],
                 [5, 26],
                 [6, 37]])

sai_1 = lambda x: 1
sai_2 = lambda x: x
sai_3 = lambda x: x**2 - 1
sai_4 = lambda x: x**3 - 3 * x
sai_5 = lambda x: x**4 - 6 * x**2 + 3
sai_6 = lambda x: 1
sai_7 = lambda x: 1
sai_8 = lambda x: 1
sai_9 = lambda x: 1
sai_10 = lambda x: 1
sai_11 = lambda x: 1

sai = lambda x: np.array([sai_1(x), sai_2(x), sai_3(x), sai_4(x), sai_5(x)])

a = [1, 1, 1, 1, 1]
phi = lambda a: sum([sai[i] * a[i] for i in range(5)])

"""k = []
for i in range(5):
    a = sai(data[i, 0]).T
    b = sai(data[i, 0])
    t = np.outer(a, b)
    k.append(t)
r = np.sum(k, axis=0)"""

G = 0.2 * np.sum([np.outer(sai(data[i, 0]).T, (sai(data[i, 0]))) for i in range(5)], axis=0)  # 本当はエルミート
A = 0.2 * np.sum([np.outer(sai(data[i, 0]).T, (sai(data[i, 1]))) for i in range(5)], axis=0)
# print(G.dot(G))
# K = np.linalg.inv(G.T.dot(G)).dot(G).dot(A)  # G：5*5、HybridKoopman Operatorだから？
# p = np.linalg.inv(G).dot(A)
K = np.linalg.pinv(G).dot(A)

mu, w, xi = la.eig(K, left=True, right=True)
B = [1, 1, 1, 1, 1]
v = (w.T.dot(B)).T  # 本当はエルミート
print(v)

true_phi = [None] * 5
for k in range(5):
    for l in range(5):
        true_phi[k] = sai(data[l, 0]).dot(xi[k])
print(true_phi)