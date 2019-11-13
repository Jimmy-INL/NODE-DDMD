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


def p_inv(X):
    X_T = torch.transpose(X, 0, 1)
    return torch.mm(torch.inverse(torch.mm(X_T, X)), X_T)  #

def Frobenius_norm(X):
    M = torch.mm(X, torch.transpose(X, 0, 1))
    return sum(torch.diag(M, 0))

width = 50
# while J(K, theta) > epsilon:
for count in range(500):
    print(count)
    for _ in range(50):
        optimizer.zero_grad()

        t = data_val[count * width:count * width + width]
        pred_sai = net(data_val[count * width:count * width + width])  # count * 50 : count * 50 + 50
        y_pred_sai = net(data_val[count * width + 1:count * width + width + 1, :])

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
        # U = torch.pinverse(G + lambda_ * I)
        # print(1111111111)
        K_tilde = torch.mm(p_inv(G + lambda_ * I), A)  # pinverseを使うとおかしくなるのでp_invで代用
        # theta = theta -
        # print(U)

        x_tilde = 0  # sum([(mu[k] ** count) * true_phi[k] * data_val[count] * v[k] for k in range(25)])
        Pred = torch.mm(K_tilde, pred_sai_T)
        Pred = torch.transpose(Pred, 0, 1)
        # y_pred_sai = y_pred_sai[0]
        y_pred_sai = torch.tensor(y_pred_sai.detach().numpy(), dtype=torch.float32)
        # res = torch.tensor(lambda_ * torch.mm(K_tilde, K_tilde), dtype=torch.float32)
        res = lambda_ * Frobenius_norm(K_tilde)

        # t = torch.transpose(pred_sai_T, 0, 1)
        # Pred = Pred.view(1, -1)
        loss = res
        for j in range(len(Pred)):
            for i in y_pred_sai[j] - Pred[j]:
                loss += torch.log(abs(i))
        # loss = loss_fn(x_tilde, data_val[count + 1, :])  # count * 50 + 1 : count * 50 + 51
        print(loss)
        loss.backward()
        optimizer.step()
