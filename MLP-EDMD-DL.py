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
l = 100
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


lambda_ = 1e-3
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
# optimizer = optim.SGD(net.parameters(), lr=1e-5)
optimizer = optim.Adam(net.parameters(), lr=1e-6)
loss_fn = nn.MSELoss()  # J(K, theta)


def p_inv(X):
    X_T = torch.transpose(X, 0, 1)
    return torch.mm(torch.inverse(torch.mm(X_T, X)), X_T)  # (X_TX)-1X_T

def Frobenius_norm(X):
    M = torch.mm(X, torch.transpose(X, 0, 1))
    return sum(torch.diag(M, 0))

width = 500
inv_N = 0.02040816
# while J(K, theta) > epsilon:
for count in range(500):
    print(count)
    for _ in range(500):
        optimizer.zero_grad()

        t = data_val[count * width:count * width + width]
        pred_sai = net(data_val[count * width:count * width + width - 1])  # count * 50 : count * 50 + 50
        y_pred_sai = net(data_val[count * width + 1:count * width + width, :])

        pred_sai_T = torch.transpose(pred_sai, 0, 1)

        G = inv_N * torch.mm(pred_sai_T, pred_sai)  # 本当はエルミート
        A = inv_N * torch.mm(pred_sai_T, y_pred_sai)
        G_np = G.detach().numpy()
        A_np = A.detach().numpy()
        K_tilde = torch.mm(p_inv(G + lambda_ * I), A)  # pinverseを使うとおかしくなるのでp_invで代用

        Pred = torch.mm(K_tilde, pred_sai_T)
        # Pred = torch.transpose(Pred, 0, 1)

        # y_pred_sai = y_pred_sai[0]
        y_pred_sai = torch.tensor(y_pred_sai, requires_grad=False)
        # y_pred_sai = torch.tensor(y_pred_sai.detach().numpy(), dtype=torch.float32)
        y_pred_sai_T = torch.transpose(y_pred_sai, 0, 1)
        # res = torch.tensor(lambda_ * torch.mm(K_tilde, K_tilde), dtype=torch.float32)
        res = lambda_ * Frobenius_norm(K_tilde)

        # t = torch.transpose(pred_sai_T, 0, 1)
        # Pred = Pred.view(1, -1)
        loss = res
        QWRETY = y_pred_sai_T - pred_sai_T
        for i in range(25):
            # loss += torch.log(sum([abs(c) for c in QWRETY[i]]))
            loss += sum([abs(c) for c in QWRETY[i]])
        """for j in range(len(Pred)):
            for i in y_pred_sai[j] - Pred[j]:
                loss += torch.log(abs(i))"""
        # loss = loss_fn(x_tilde, data_val[count + 1, :])  # count * 50 + 1 : count * 50 + 51
        # loss = loss_fn(pred_sai, y_pred_sai)
        # loss = loss_fn(Pred, y_pred_sai_T)
        # loss =torch.tensor(1, requires_grad=True)
        print("loss", loss)
        # print(net.parameters().item())
        loss.backward()
        optimizer.step()

        """E_reconを計算"""
        K = np.linalg.pinv(G_np).dot(A_np)
        mu, w, xi = la.eig(K, left=True, right=True)
        B = [1] * 25
        v = (w.T.dot(B)).T  # 本当はエルミート
        # print(v)
        # pred_sai_2 = np.reshape(pred_sai, (3, 4))
        true_phi = [0] * 25
        pred_sai_np = pred_sai.detach().numpy()
        for k in range(25):
            p = xi[k]
            q = pred_sai_np
            true_phi[k] += xi[k].dot(pred_sai_np[0])
        #print(true_phi)

        x_tilde = [0] * (width - 1)
        for n in range(width - 1):
            x_tilde[n] = sum([(mu[k] ** (n + 1)) * true_phi[k] * v[k] for k in range(25)])  # sum([(mu[k] ** count) * true_phi[k] * data_val[count] * v[k] for k in range(25)])

        data_val_np = data_val.numpy()
        k = data_val_np[0][0] - x_tilde[n]
        #print(k)
        E_recon = np.sqrt(inv_N * sum([abs(data_val_np[n][0] - x_tilde[n]) ** 2 + abs(data_val_np[n][1] - x_tilde[n]) ** 2
                                       for n in range(width - 1)]))
        #print(E_recon)

        """E_eigfunc_jを計算"""
        """
        E_eigfunc = [0] * 25
        for j in range(25):
            E_eigfunc[j] = np.sqrt(inv_N * sum([abs(data_val[n + 1][0] - mu[j] * x_tilde[n][0]) ** 2 + abs(data_val[n + 1][1] - x_tilde[n][1]) ** 2
                                           for n in range(width - 1)]))"""

