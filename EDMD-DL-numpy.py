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
I = np.eye(25, 25)
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


count = 0
# while J(K, theta) > epsilon:
for _ in range(50):
    optimizer.zero_grad()

    t = data_val[count:count + 1]
    pred_sai = net(data_val[count:count + 1])  # count * 50 : count * 50 + 50
    y_pred_sai = net(data_val[count + 1:count + 2, :])

    print(y_pred_sai.grad_fn)
    pred_sai = pred_sai.detach().numpy()
    y_pred_sai = y_pred_sai.detach().numpy()
    G = np.outer(pred_sai.T, pred_sai)  # 本当はエルミート
    Q = pred_sai.T.dot(pred_sai)
    A = np.outer(pred_sai.T, y_pred_sai)
    # print(G.dot(G))
    # K = np.linalg.inv(G.T.dot(G)).dot(G).dot(A)  # G：5*5、HybridKoopman Operatorだから？
    # p = np.linalg.inv(G).dot(A)
    K = np.linalg.pinv(G).dot(A)

    mu, w, xi = la.eig(K, left=True, right=True)
    B = [1] * 25
    v = (w.T.dot(B)).T  # 本当はエルミート
    #print(v)

    # pred_sai_2 = np.reshape(pred_sai, (3, 4))
    true_phi = [0] * 25
    for k in range(25):
        p = xi[k]
        q = pred_sai
        true_phi[k] += xi[k].dot(pred_sai[0])
    print(true_phi)
    K_tilde = np.linalg.pinv(G + lambda_ * I).dot(A)
    # theta = theta -

    x_tilde = 0 # sum([(mu[k] ** count) * true_phi[k] * data_val[count] * v[k] for k in range(25)])
    Pred = K_tilde.dot(pred_sai[0])
    Pred = torch.tensor(Pred, dtype=torch.float32, requires_grad=True)
    y_pred_sai = y_pred_sai[0]
    y_pred_sai = torch.tensor(y_pred_sai, dtype=torch.float32)
    res = torch.tensor(lambda_ * K_tilde.dot(K_tilde), dtype=torch.float32)

    print(y_pred_sai.grad_fn)
    loss = loss_fn(Pred, y_pred_sai)  # + res
    # loss = loss_fn(x_tilde, data_val[count + 1, :])  # count * 50 + 1 : count * 50 + 51
    loss.backward()
    optimizer.step()
    count += 1