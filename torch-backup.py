import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from scipy import linalg as la

d = 2
l = 20
M = 25  # 22


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(d, l),
            nn.Tanh(),
            nn.Linear(l, l),
            nn.Tanh(),
            nn.Linear(l, l),
            nn.Tanh(),
            nn.Linear(l, M),
        )


params = {}
params['data_name'] = 'Duffing_oscillator'
data_val = np.loadtxt(('./data/%s_val_x.csv' % (params['data_name'])), delimiter=',', dtype=np.float64)  # ここでデータを読み込む
data_val = torch.tensor(data_val, dtype=torch.float32)


def J(K, theta):
    pass


lambda_ = 0.001
I = torch.tensor(np.eye(25, 25), dtype=torch.float32)
# K_tilde = np.linalg.pinv(G + lambda_.dot(I)).dot(A)
epsilon = 0.1

net = nn.Sequential(
    nn.Linear(d, l),
    nn.Tanh(),
    nn.Linear(l, l),
    nn.Tanh(),
    nn.Linear(l, l),
    nn.Tanh(),
    nn.Linear(l, M),
)
optimizer = optim.SGD(net.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()  # J(K, theta)


def inverse(matrix):
    def determinant(matrix):
        l = len(matrix)
        u = [0 for r in range(0, l)]
        d = 1

        def reduct(matrix):
            w = [[0 for j in range(len(matrix[0]) - 1)] for k in range(len(matrix) - 1)]
            for row in range(1, len(matrix[0])):
                for column in range(1, len(matrix)):
                    s = matrix[column][row] - (matrix[column][0] / matrix[0][0]) * matrix[0][row]
                    w[column - 1][row - 1] = s
            return w

        def loop(matrix):
            if len(matrix) > 1:
                u[-len(matrix)] = matrix[0][0]
                matrix = reduct(matrix)
            elif len(matrix) == 1:
                u[-1] = matrix[0][0]
            else:
                print("error")
            return matrix

        while l > 0:
            matrix = loop(matrix)
            l = l - 1

        for ele in range(0, len(u)):
            d = d * u[ele]
        return d

    def coef(matrix, row, column):
        u = [[0 for k in range(len(matrix[0]) - 1)] for l in range(len(matrix) - 1)]
        for j in range(0, len(matrix[0])):
            for i in range(0, len(matrix)):
                if (j < (row - 1)) and (i < (column - 1)):
                    u[i][j] = matrix[i][j]
                elif (j < (row - 1)) and (i > (column - 1)):
                    u[i - 1][j] = matrix[i][j]
                elif (j > (row - 1)) and (i < (column - 1)):
                    u[i][j - 1] = matrix[i][j]
                elif (j > (row - 1)) and (i > (column - 1)):
                    u[i - 1][j - 1] = matrix[i][j]
                else:
                    pass
        return u

    inv = [[0 for i in range(0, len(matrix[0]))] for j in range(0, len(matrix))]

    for column in range(1, len(inv[0]) + 1):
        for row in range(1, len(inv) + 1):
            inv[row - 1][column - 1] = ((-1) ** (row + column)) * determinant(coef(matrix, row, column)) / determinant(
                matrix)

    return inv


def p_inv(X):
    X_T = torch.transpose(X, 0, 1)
    return torch.mm(torch.inverse(torch.mm(X_T, X)), X_T)  #

def Frobenius_norm(X):
    M = torch.mm(X, torch.transpose(X, 0, 1))
    return sum(torch.diag(M, 0))

count = 0
# while J(K, theta) > epsilon:
for _ in range(50):
    optimizer.zero_grad()

    t = data_val[count:count + 1]
    pred_sai = net(data_val[count:count + 1])  # count * 50 : count * 50 + 50
    y_pred_sai = net(data_val[count + 1:count + 2, :])

    # pred_sai = pred_sai.detach().numpy()
    # y_pred_sai = y_pred_sai.detach().numpy()
    pred_sai_T = torch.transpose(pred_sai, 0, 1)

    G = torch.mm(pred_sai_T, pred_sai)  # 本当はエルミート
    A = torch.mm(pred_sai_T, y_pred_sai)
    # print(G.dot(G))
    # K = np.linalg.inv(G.T.dot(G)).dot(G).dot(A)  # G：5*5、HybridKoopman Operatorだから？
    # p = np.linalg.inv(G).dot(A)
    # K = np.linalg.pinv(G).dot(A)

    """mu, w, xi = la.eig(K, left=True, right=True)
    B = [1] * 25
    v = (w.T.dot(B)).T  # 本当はエルミート
    # print(v)

    # pred_sai_2 = np.reshape(pred_sai, (3, 4))
    true_phi = [0] * 25
    for k in range(25):
        p = xi[k]
        q = pred_sai
        true_phi[k] += xi[k].dot(pred_sai[0])
    print(true_phi)"""

    # K_tilde = np.linalg.pinv(G + lambda_ * I).dot(A)
    R = G + lambda_ * I
    # U = torch.pinverse(G + lambda_ * I)
    # print(1111111111)
    K_tilde = torch.mm(p_inv(G + lambda_ * I), A)  # pinverseを使うとおかしくなるのでp_invで代用
    # theta = theta -
    # print(U)

    x_tilde = 0  # sum([(mu[k] ** count) * true_phi[k] * data_val[count] * v[k] for k in range(25)])
    Pred = torch.mm(K_tilde, pred_sai_T)
    # y_pred_sai = y_pred_sai[0]
    y_pred_sai = torch.tensor(y_pred_sai.detach().numpy(), dtype=torch.float32)
    # res = torch.tensor(lambda_ * torch.mm(K_tilde, K_tilde), dtype=torch.float32)
    res = lambda_ * Frobenius_norm(K_tilde)

    # t = torch.transpose(pred_sai_T, 0, 1)
    Pred = Pred.view(1, -1)
    loss = res
    for j in range(len(Pred)):
        for i in y_pred_sai[j] - Pred[j]:
            loss += torch.log(abs(i))
    #loss = sum([sum([abs(i) for i in y_pred_sai[j] - Pred[j]]) for j in range(len(Pred))])  # + res
    # loss = loss_fn(x_tilde, data_val[count + 1, :])  # count * 50 + 1 : count * 50 + 51
    print(loss)
    loss.backward()
    optimizer.step()
    count += 1
