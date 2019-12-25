import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from scipy import linalg as la

from matplotlib import pyplot as plt
from scipy.stats import uniform


data_name = 'spectrum'  # 'Duffing_oscillator', 'Linear'，Discrete_Linear Duffing_oscillator spectrum-1

d = 2
l = 200  # 70
M = 22  # 22
I = torch.eye(M + 3, M + 3)

N = 100000
#width = 11  #11
inv_N = 1/N  #0.1


def data_Preprocessing(tr_val_te):
    data = np.loadtxt(('./data/%s_%s.csv' % (data_name, tr_val_te)), delimiter=',', dtype=np.float64)[:N]
    # np.loadtxt(('./data/%s_val_x.csv' % (params['data_name'])), delimiter=',', dtype=np.float64)  # ここでデータを読み込む
    #data = torch.tensor(data, dtype=torch.float32)
    return data

def p_inv(X):
    X_T = torch.transpose(X, 0, 1)
    return torch.mm(torch.inverse(torch.mm(X_T, X)), X_T)  # (X_TX)-1X_T

def Frobenius_norm(X):
    M = torch.mm(X, torch.transpose(X, 0, 1))
    return torch.sum(torch.diag(M, 0))

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
        plt.scatter([i for i in range(50)], predict, label="predict", color="orange")  # 予測，オレンジ
        #plt.plot(predict, label="predict")  # 予測，オレンジ
        #plt.plot(phi_predict, label="predict")  # 予測Φ，緑
        plt.title("x2_trajectory")
        plt.xlabel('n')
        plt.ylabel('x2')
        plt.legend()
    plt.savefig("png/" + name + ".png")
    plt.savefig("eps/" + name + ".eps")
    plt.show()



# while J(K, theta) > epsilon:
x = []
y = []
X = []
Y = []
count = 0

"""netを学習"""
count = 0
rotation = 0
x = [i for i in range(rotation)]


"""学習済みのnetを使って，E_reconを計算"""





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


sai = lambda x, y: np.array([sai_1(x, y), sai_2(x, y), sai_3(x, y), sai_4(x, y), sai_5(x, y)
                             , sai_6(x, y), sai_7(x, y), sai_8(x, y), sai_9(x, y), sai_10(x, y)
                             , sai_11(x, y), sai_12(x, y), sai_13(x, y), sai_14(x, y), sai_15(x, y)
                             , sai_16(x, y), sai_17(x, y), sai_18(x, y), sai_19(x, y), sai_20(x, y)
                             , sai_21(x, y), sai_22(x, y), sai_23(x, y), sai_24(x, y), sai_25(x, y)])


x_data = data_Preprocessing("train_x")
y_data = data_Preprocessing("train_y")

G = 1 / N * np.sum([np.outer(sai(x_data[i, 0], x_data[i, 1]).T, (sai(x_data[i, 0], x_data[i, 1]))) for i in range(N)], axis=0)  # 本当はエルミート
A = 1 / N * np.sum([np.outer(sai(x_data[i, 0], x_data[i, 1]).T, (sai(y_data[i, 0], y_data[i, 1]))) for i in range(N)], axis=0)


K = np.linalg.pinv(G).dot(A)
mu, w, xi = la.eig(K, left=True, right=True)

print(K)
mu_real = [i.real for i in mu]
mu_imag = [i.imag for i in mu]
graph(mu_real, mu_imag, "eigenvalue", "scatter")










mu = 0
for tr_val_te in ["E_recon_50"]:
    #B = torch.tensor([[1 if ((i == 22 and j == 0) or (i == 23 and j == 1)) else 0 for i in range(M + 3)] for j in range(2)])
    B = torch.tensor(
        [[1 / 2 if ((i == 1 and j == 0) or (i == 5 and j == 1)) else 0 for i in range(M + 3)] for j in range(2)])
    B = B.detach().numpy()
    #K = K.detach().numpy()

    data = data_Preprocessing("E_recon_50")
    width = 50
    mu, xi, zeta = la.eig(K, left=True, right=True)

    # xi内積zetaでxiを正規化
    confirm = np.conjugate(xi.T).dot(zeta)
    print(np.diag(confirm))
    adjustment = np.diag(confirm)

    for i in range(M + 3):
        for j in range(M + 3):
            y = adjustment[i]
            xi[j][i] = xi[j][i] / np.conjugate(adjustment[i])

    confirm = np.conjugate(xi.T).dot(zeta)
    # print(np.diag(confirm))

    #xi = np.conjugate(xi)
    """"# mu zeta = K zeta
    # mu z = K.T z
    # mu z.T = z.T K，xi = z.T
    mu2, _, z = la.eig(K.T, left=True, right=True)
    # print(mu[0:20])
    # print(mu2[0:20])
    # print(mu2[1] * z[:, 1].T - z[:, 1].T.dot(K))

    print(mu[1] * zeta[:, 1] - K.dot(zeta[:, 1]))
    print(mu[1] * xi[:, 1].T - xi[:, 1].T.dot(K))
    print(mu[1] * zeta[:, 1], K.dot(zeta[:, 1]))
    print(mu[1] * xi[:, 1].T, xi[:, 1].T.dot(K))"""

    mu_real = [i.real for i in mu]
    mu_imag = [i.imag for i in mu]
    graph(mu_real, mu_imag, "eigenvalue", "scatter")


    while count < 10:
        """x_data = data[count * width:count * width + width]  # N = 10
        sai = net(x_data)
        fixed_sai = torch.tensor([i + [0.1] for i in x_data.detach().tolist()], dtype=torch.float32)
        sai = torch.cat([sai, fixed_sai], dim=1).detach().numpy()
        sai_T = sai.T"""

        x_data = data[count * width:count * width + width]  # N = 10
        Sai = sai(x_data[0, 0], x_data[0, 1]).reshape(1, 25)
        for i in range(1, width):
            tmp = sai(x_data[i, 0], x_data[i, 1])
            Sai = np.vstack((Sai, tmp))
        sai_T = Sai.T

        """E_reconを計算"""
        #m = B.dot(zeta)  # (xi.T.dot(B)).T  # 本当はエルミート
        #m = m.T
        m = xi.T.dot(B.T)

        # sai_T = torch.rand(M + 3, width - 1) * 100
        #phi = (xi.T).dot(sai_T)
        phi = Sai.dot(zeta)
        phi = phi.T

        for kk in range(M + 3):
            print("-----------------------", kk, "--------------------------------------")
            for i in range(1, width):
                # print(Phi[10][i - 1] / Phi[10][i], Mu[10])
                print(phi[kk][i] / phi[kk][i - 1], mu[kk])

        x_tilde = [[0, 0] for _ in range(width)]  # [[0, 0]] * (width - 1)
        x_tilde_phi = [[0, 0] for _ in range(width)]
        x_tilde[0][0] = float(x_data[0][0])
        x_tilde[0][1] = float(x_data[0][1])
        x_tilde_phi[0][0] = float(x_data[0][0])
        x_tilde_phi[0][1] = float(x_data[0][1])
        for n in range(1, width):
            print((mu[1] ** n) * phi[1][0], phi[1][n])
            x_tilde[n][0] = sum([(mu[k] ** n) * phi[k][0] * m[k][0] for k in range(M + 3)]).real  # sum([(mu[k] ** count) * true_phi[k] * data_val[count] * v[k] for k in range(25)])
            x_tilde[n][1] = sum([(mu[k] ** n) * phi[k][0] * m[k][1] for k in range(M + 3)]).real

            x_tilde_phi[n][0] = sum([phi[k][n] * m[k][0] for k in range(M + 3)]).real  # sum([(mu[k] ** count) * true_phi[k] * data_val[count] * v[k] for k in range(25)])
            x_tilde_phi[n][1] = sum([phi[k][n] * m[k][1] for k in range(M + 3)]).real

        #x_data = x_data.detach().numpy()
        E_recon = (inv_N * sum([abs(x_data[n][0] - x_tilde[n][0]) ** 2 + abs(x_data[n][1] - x_tilde[n][1]) ** 2
                                for n in range(width)])) ** 0.5
        print("E_recon", E_recon)

        count += 1
        x_tilde_0 = [j for i, j in x_tilde]
        x_tilde_phi_0 = [j for i, j in x_tilde_phi]

        graph([], [], "x2_traj_" + "{stp:02}".format(stp=count), "multi_plot"
              , x_data[:, 1], x_tilde_0, x_tilde_phi_0)


"""学習済みのnetを使って，E_eigfuncを計算"""
I_number = 10
data = data_Preprocessing("E_eigfunc_confirm")
width = 2
phi_list = [[0 for count in range(I_number)] for j in range(25)]
y_phi_list = [[0 for count in range(I_number)] for j in range(25)]

for count in range(I_number):
    x_data = data[count * width:count * width + width]
    pred_sai = net(x_data)  # count * 50 : count * 50 + 50
    fixed_sai = torch.tensor([i + [0.1] for i in x_data.detach().tolist()], dtype=torch.float32)
    pred_sai = torch.cat([pred_sai, fixed_sai], dim=1).detach().numpy()
    pred_phi = (xi.T).dot(pred_sai.T)

    for j in range(M + 3):
        phi_list[j][count] = pred_phi[j][0]
        y_phi_list[j][count] = pred_phi[j][1]

    """for j in range(M + 3):
        if j < 22:
            phi_list[j][count] = pred_sai[0][j].detach().numpy()
            y_phi_list[j][count] = pred_sai[1][j].detach().numpy()
        elif j == M:
            phi_list[j][count] = x_data[0][0].detach().numpy()
            y_phi_list[j][count] = x_data[1][0].detach().numpy()
        elif j == M + 1:
            phi_list[j][count] = x_data[0][1].detach().numpy()
            y_phi_list[j][count] = x_data[1][1].detach().numpy()
        elif j == M + 2:
            phi_list[j][count] = 0.1
            y_phi_list[j][count] = 0.1"""


"""E_eigfunc_jを計算"""

# N = 100000
a = -100
b = 100
# x = uniform(loc=a, scale=b-a).rvs(size=N)
E_eigfunc = [0] * (M + 3)
for j in range(M + 3):
    tmp = [(y_phi_list[j][count] - mu[j] * phi_list[j][count]).real ** 2
           + (y_phi_list[j][count] - mu[j] * phi_list[j][count]).imag ** 2
                                   for count in range(I_number)]
    E_eigfunc[j] = np.sqrt(1 / I_number * sum(tmp))
    print("E_eigfunc", E_eigfunc[j])