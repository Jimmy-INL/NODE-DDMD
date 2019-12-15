import numpy as np
from scipy import linalg as la
# DMD Algorithm

import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import dot, multiply, diag, power
from numpy import pi, exp, sin, cos, cosh, tanh, real, imag
from numpy.linalg import inv, eig, pinv
from scipy.linalg import svd, svdvals
from scipy.integrate import odeint, ode, complex_ode
from warnings import warn

# define time and space domains
x = np.linspace(-10, 10, 100)
t = np.linspace(0, 6 * pi, 80)
dt = t[2] - t[1]
Xm, Tm = np.meshgrid(x, t)

A12 = np.array([[-1, 1, 0],
                [0, -1, 1],
                [1, 0, -1]])  # Unknown head nodes incidence matrix
A10 = 3



# Extended Dynamic Mode Decomposition
sai_1 = lambda x, y: 1
sai_2 = lambda x, y: 2 * x
sai_3 = lambda x, y: 4 * x**2 - 2
sai_4 = lambda x, y: 8 * x**3 - 12 * x
sai_5 = lambda x, y: 16 * x**4 - 48 * x**2 + 12
sai_6 = lambda x, y: 1 * (2 * y)
sai_7 = lambda x, y: 2 * x * (2 * y)
sai_8 = lambda x, y: (4 * x**2 - 2) * (2 * y)
sai_9 = lambda x, y: (8 * x**3 - 12 * x) * (2 * y)
sai_10 = lambda x, y: (16 * x**4 - 48 * x**2 + 12) * (2 * y)
sai_11 = lambda x, y: 1 * (4 * y**2 - 2)
sai_12 = lambda x, y: 2 * x * (4 * y**2 - 2)
sai_13 = lambda x, y: (4 * x**2 - 2) * (4 * y**2 - 2)
sai_14 = lambda x, y: (8 * x**3 - 12 * x) * (4 * y**2 - 2)
sai_15 = lambda x, y: (16 * x**4 - 48 * x**2 + 12) * (4 * y**2 - 2)
sai_16 = lambda x, y: 1 * (8 * y**3 - 12)
sai_17 = lambda x, y: 2 * x * (8 * y**3 - 12)
sai_18 = lambda x, y: (4 * x**2 - 2) * (8 * y**3 - 12)
sai_19 = lambda x, y: (8 * x**3 - 12 * x) * (8 * y**3 - 12)
sai_20 = lambda x, y: (16 * x**4 - 48 * x**2 + 12) * (8 * y**3 - 12)
sai_21 = lambda x, y: 1 * (16 * y**4 - 48 * y**2 + 12)
sai_22 = lambda x, y: 2 * x * (16 * y**4 - 48 * y**2 + 12)
sai_23 = lambda x, y: (4 * x**2 - 2) * (16 * y**4 - 48 * y**2 + 12)
sai_24 = lambda x, y: (8 * x**3 - 12 * x) * (16 * y**4 - 48 * y**2 + 12)
sai_25 = lambda x, y: (16 * x**4 - 48 * x**2 + 12) * (16 * y**4 - 48 * y**2 + 12)
# Testing the concept
data = np.array([[1, 2],
                 [2, 5],
                 [3, 10],
                 [4, 17],
                 [5, 26],
                 [6, 37]])


"""sai_2 = lambda x: x
sai_3 = lambda x: x**2 - 1
sai_4 = lambda x: x**3 - 3 * x
sai_5 = lambda x: x**4 - 6 * x**2 + 3
sai_6 = lambda x, y: 32 * x**5 - 160 * x**3 + 120 * x
sai_7 = lambda x, y: 64 * x**6 - 480 * x**4 + 720 * x**2 - 120"""


sai = lambda x, y: np.array([sai_1(x, y), sai_2(x, y), sai_3(x, y), sai_4(x, y), sai_5(x, y)
                             , sai_6(x, y), sai_7(x, y), sai_8(x, y), sai_9(x, y), sai_10(x, y)
                             , sai_11(x, y), sai_12(x, y), sai_13(x, y), sai_14(x, y), sai_15(x, y)
                             , sai_16(x, y), sai_17(x, y), sai_18(x, y), sai_19(x, y), sai_20(x, y)
                             , sai_21(x, y), sai_22(x, y), sai_23(x, y), sai_24(x, y), sai_25(x, y)])

a = [1, 1, 1, 1, 1, 1]
#phi = lambda a: sum([sai[i] * a[i] for i in range(6)])

"""k = []
for i in range(5):
    a = sai(data[i, 0]).T
    b = sai(data[i, 0])
    t = np.outer(a, b)
    k.append(t)
r = np.sum(k, axis=0)"""
N = 10000
data_name = 'Discrete_Linear'  # 'Duffing_oscillator' , 'Linear'
def data_Preprocessing(tr_val_te):
    data = np.loadtxt(('./data/%s_%s.csv' % (data_name, tr_val_te)), delimiter=',', dtype=np.float64)[:N]
    return data

#グラフ
def graph(x, y, name, type, correct=[], predict=[], phi_predict=[]):  # plt.xlim(1300,)
    plt.figure()
    if type == "plot":
        plt.plot(x, y)
        plt.title('MLP Model Loss')  # タイトル名
        plt.xlabel('epoch')
        plt.ylabel('loss')
    elif type == "scatter":
        plt.scatter(x, y)
        plt.title('Eigenvalue')  # タイトル名
        plt.xlabel('Re(μ)')
        plt.ylabel('Im(μ)')
        plt.grid(True)  # 目盛の表示
    elif type == "multi_plot":
        plt.plot(correct, label="correct")  # 実データ，青
        plt.plot(predict, label="predict")  # 予測，オレンジ
        #plt.plot(phi_predict, label="predict")  # 予測Φ，緑
        plt.title("x1_trajectory")
        plt.xlabel('n')
        plt.ylabel('x1')
        plt.legend()
    plt.savefig("png/" + name + ".png")
    plt.savefig("eps/" + name + ".eps")
    plt.show()


x_data = data_Preprocessing("train_x")
y_data = data_Preprocessing("train_y")

# G = 1 / N * np.sum([np.outer(sai(data[i, 0]).T, (sai(data[i, 0]))) for i in range(N)], axis=0)  # 本当はエルミート
# A = 1 / N * np.sum([np.outer(sai(data[i, 0]).T, (sai(data[i, 1]))) for i in range(N)], axis=0)
# print(G.dot(G))
# K = np.linalg.inv(G.T.dot(G)).dot(G).dot(A)  # G：5*5、HybridKoopman Operatorだから？
# p = np.linalg.inv(G).dot(A)
G = 1 / N * np.sum([np.outer(sai(x_data[i, 0], x_data[i, 1]).T, (sai(x_data[i, 0], x_data[i, 1]))) for i in range(N)], axis=0)  # 本当はエルミート
A = 1 / N * np.sum([np.outer(sai(x_data[i, 0], x_data[i, 1]).T, (sai(y_data[i, 0], y_data[i, 1]))) for i in range(N)], axis=0)
# print(sai(x_data[2, 0], x_data[2, 1]).T, (sai(y_data[2, 0], y_data[2, 1])))
# k = sai(x_data[2, 0], x_data[2, 1]).T
# p = sai(x_data[2, 0], x_data[2, 1])
# G = inv_N * torch.mm(sai(x_data[i, 0], x_data[i, 1]).T, sai(x_data[i, 0], x_data[i, 1]))  # 本当はエルミート
# A = inv_N * torch.mm(sai(x_data[i, 0], x_data[i, 1]).T, sai(y_data[i, 0], y_data[i, 1]))

K = np.linalg.pinv(G).dot(A)

mu, w, xi = la.eig(K, left=True, right=True)

mu_real = [i.real for i in mu]
mu_imag = [i.imag for i in mu]
graph(mu_real, mu_imag, "eigenvalue", "scatter")

"""B = [1, 1, 1, 1, 1]
v = (w.T.dot(B)).T  # 本当はエルミート
print(v)

true_phi = [None] * 5
for k in range(5):
    for l in range(5):
        true_phi[k] = sai(data[l, 0]).dot(xi[k])
print(true_phi)"""

"""学習済みのnetを使って，E_reconを計算"""

M = 22
data = data_Preprocessing("train_x")
count = 0
width = 10
"""Bを計算，X=BΨ"""
X25 = data[0].reshape(1,2)
for i in range(1, M + 3):
    tmp = x_data[i]
    X25 = np.vstack((X25, tmp))

rrr = np.array(sai(x_data[0, 0], x_data[0, 1])).reshape(1,25)
for i in range(1, M + 3):
    tmp = np.array(sai(x_data[i, 0], x_data[i, 1]))
    rrr = np.vstack((rrr, tmp))


B = X25.T.dot(np.linalg.inv(rrr.T))
data = data_Preprocessing("train_x")  # E_recon_50
width = 10

mu, xi, zeta = la.eig(K, left=True, right=True)

while count < 99:
    x_data = data[count * width:count * width + width]  # N = 10
    Sai = sai(x_data[0, 0], x_data[0, 1]).reshape(1, 25)
    for i in range(1, width):
        tmp = sai(x_data[i, 0], x_data[i, 1])
        Sai = np.vstack((Sai, tmp))
    sai_T = Sai.T

    """E_reconを計算"""
    m = B.dot(zeta)  # (xi.T.dot(B)).T  # 本当はエルミート
    m = m.T
    # sai_T = torch.rand(M + 3, width - 1) * 100
    phi = (xi.T).dot(sai_T)

    x_tilde = [[0, 0] for _ in range(width)]  # [[0, 0]] * (width - 1)
    x_tilde_phi = [[0, 0] for _ in range(width)]
    x_tilde[0][0] = x_data[0][0]
    x_tilde[0][1] = x_data[0][1]
    x_tilde_phi[0][0] = x_data[0][0]
    x_tilde_phi[0][1] = x_data[0][1]
    for n in range(1, width):
        print((mu[1] ** n) * phi[1][0], phi[1][n])
        x_tilde[n][0] = sum([(mu[k] ** n) * phi[k][0] * m[k][0] for k in range(M + 3)]).real  # sum([(mu[k] ** count) * true_phi[k] * data_val[count] * v[k] for k in range(25)])
        x_tilde[n][1] = sum([(mu[k] ** n) * phi[k][0] * m[k][1] for k in range(M + 3)]).real

        x_tilde_phi[n][0] = sum([phi[k][n] * m[k][0] for k in range(M + 3)]).real  # sum([(mu[k] ** count) * true_phi[k] * data_val[count] * v[k] for k in range(25)])
        x_tilde_phi[n][1] = sum([phi[k][n] * m[k][1] for k in range(M + 3)]).real

    E_recon = (1 / N * sum([abs(x_data[n][0] - x_tilde[n][0]) ** 2 + abs(x_data[n][1] - x_tilde[n][1]) ** 2
                            for n in range(width)])) ** 0.5
    print("E_recon", E_recon)

    count += 1
    x_tilde_0 = [i for i, j in x_tilde]
    x_tilde_phi_0 = [i for i, j in x_tilde_phi]

    graph([], [], "x1_traj_" + "{stp:02}".format(stp=count), "multi_plot"
          , x_data[:, 0], x_tilde_0, x_tilde_phi_0)
