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

params = {}
params['data_name'] = 'Duffing_oscillator'



def J(K, theta):
    pass


lambda_ = 0.1
I = torch.tensor(np.eye(25, 25), dtype=torch.float32)
# K_tilde = np.linalg.pinv(G + lambda_.dot(I)).dot(A)
epsilon = 0.1

d = 2
l = 100
M = 22  # 22

net = nn.Sequential(
    nn.Linear(d, l),
    nn.Tanh(),
    nn.Linear(l, l),
    nn.Tanh(),
    #nn.Dropout(0.5),
    nn.Linear(l, l),
    nn.Tanh(),
    nn.Linear(l, M),
)
optimizer = optim.SGD(net.parameters(), lr=1e-3)
# optimizer = optim.Adam(net.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()  # J(K, theta)

def data_Preprocessing(tr_val_te):
    data = np.loadtxt(('./data/%s_%s_x.csv' % (params['data_name'], tr_val_te)), delimiter=',', dtype=np.float64)
    # np.loadtxt(('./data/%s_val_x.csv' % (params['data_name'])), delimiter=',', dtype=np.float64)  # ここでデータを読み込む
    data = torch.tensor(data, dtype=torch.float32)
    return data

def p_inv(X):
    X_T = torch.transpose(X, 0, 1)
    return torch.mm(torch.inverse(torch.mm(X_T, X)), X_T)  # (X_TX)-1X_T

def Frobenius_norm(X):
    M = torch.mm(X, torch.transpose(X, 0, 1))
    return sum(torch.diag(M, 0))

def graph(y, st):
    plots = plt.plot(y)
    plt.legend(plots, st,  # 3つのプロットラベルの設定
               loc='best',  # 線が隠れない位置の指定
               framealpha=0.25,  # 凡例の透明度
               prop={'size': 'small', 'family': 'monospace'})  # 凡例のfontプロパティ
    plt.title('Data Graph')  # タイトル名
    plt.xlabel('count')  # 横軸のラベル名
    plt.ylabel('loss')  # 縦軸のラベル名
    plt.grid(True)  # 目盛の表示
    plt.tight_layout()  # 全てのプロット要素を図ボックスに収める
    # 描画実行
    plt.show()


width = 11
inv_N = 0.1
# while J(K, theta) > epsilon:
x = []
y = []
X = []
Y = []
count = 0
K_tilde = []

"""netを学習"""
data = data_Preprocessing("train")
# if tr_val_te != "train":
count = 0
for _ in range(1):
    while count < 1000:
        optimizer.zero_grad()

        x_data = data[count * width:count * width + width - 1]
        y_data = data[count * width + 1:count * width + width]  # data[count * width + 1:count * width + width, :]
        pred_sai = net(x_data)  # count * 50 : count * 50 + 50
        y_pred_sai = net(y_data)

        fixed_sai = torch.tensor([i + [0.1] for i in x_data.detach().tolist()], dtype=torch.float32)
        pred_sai = torch.cat([pred_sai, fixed_sai], dim=1)
        y_fixed_sai = torch.tensor([i + [0.1] for i in y_data.detach().tolist()], dtype=torch.float32)
        y_pred_sai = torch.cat([y_pred_sai, y_fixed_sai], dim=1)

        pred_sai_T = torch.transpose(pred_sai, 0, 1)

        G = inv_N * torch.mm(pred_sai_T, pred_sai)  # 本当はエルミート
        A = inv_N * torch.mm(pred_sai_T, y_pred_sai)

        # K_tilde = torch.mm(p_inv(G + lambda_ * I), A)  # pinverseを使うとおかしくなるのでp_invで代用
        K_tilde = torch.mm(torch.inverse(G + lambda_ * I), A)
        K_tilde = torch.tensor(K_tilde, requires_grad=False)

        Pred = torch.mm(K_tilde, pred_sai_T)
        # Pred = torch.transpose(Pred, 0, 1)

        # y_pred_sai = y_pred_sai[0]
        # y_pred_sai = torch.tensor(y_pred_sai)
        # y_pred_sai = torch.tensor(y_pred_sai.detach().numpy(), dtype=torch.float32)
        y_pred_sai_T = torch.transpose(y_pred_sai, 0, 1)
        # res = torch.tensor(lambda_ * torch.mm(K_tilde, K_tilde), dtype=torch.float32)
        res = lambda_ * Frobenius_norm(K_tilde)

        # t = torch.transpose(pred_sai_T, 0, 1)
        # Pred = Pred.view(1, -1)
        loss = res
        QWRETY = y_pred_sai_T - Pred
        for i in range(25):
            # loss += torch.log(sum([abs(c) for c in QWRETY[i]]))
            loss += sum([c ** 2 for c in QWRETY[i]])
        """for j in range(len(Pred)):
            for i in y_pred_sai[j] - Pred[j]:
                loss += torch.log(abs(i))"""
        # loss = loss_fn(x_tilde, data_val[count + 1, :])  # count * 50 + 1 : count * 50 + 51
        # loss = loss_fn(pred_sai, y_pred_sai)
        # loss = loss_fn(Pred, y_pred_sai_T)
        # loss =torch.tensor(1, requires_grad=True)
        # x.append(rout)
        # if loss < 1.5:
        y.append(loss)
        print("loss", loss)
        # print(net.parameters().item())
        loss.backward()
        optimizer.step()

        count += 1
    graph(y, "train")
    count = 0


"""学習済みのnetを使って，E_reconを計算"""
K = K_tilde
mu = 0
for tr_val_te in ["train"]:
    data = data_Preprocessing(tr_val_te)
    count = 0
    """Bを計算"""
    X25 = data[0].view(2, -1)
    for i in range(1, 25):
        x2_data = data[11 * i].view(2, -1)
        X25 = torch.cat([X25, x2_data], dim=1)

    Sai = net(data[0])
    tmp = data[0].detach().tolist() + [0.1]  # [i + [0.1] for i in data[0].detach().tolist()]
    fixed_sai1 = torch.tensor(tmp, dtype=torch.float32)
    Sai = torch.cat([Sai, fixed_sai1])
    Sai = Sai.view(25, -1)
    for i in range(1, 25):
        sai = net(data[11 * i])
        tmp = data[11 * i].detach().tolist() + [0.1]
        fixed_sai = torch.tensor(tmp, dtype=torch.float32)
        sai = torch.cat([sai, fixed_sai])
        sai = sai.view(25, -1)
        Sai = torch.cat([Sai, sai], dim=1)

    # Sai = torch.transpose(Sai, 0, 1)
    B = torch.mm(X25, torch.inverse(Sai))
    B = B.detach().numpy()

    while count < 10:
        x_data = data[count * width:count * width + width - 1]  # N = 10
        sai = net(x_data)
        fixed_sai = torch.tensor([i + [0.1] for i in x_data.detach().tolist()], dtype=torch.float32)
        sai = torch.cat([sai, fixed_sai], dim=1).detach().numpy()
        sai_T = sai.T
        """E_reconを計算"""
        mu, xi, zeta = la.eig(K, left=True, right=True)

        m = B.dot(zeta)  # (xi.T.dot(B)).T  # 本当はエルミート
        m = m.T
        phi = xi.T.dot(sai_T)

        x_tilde = [[0, 0]] * (width - 1)
        for n in range(width - 1):
            x_tilde[n] = sum([(mu[k] ** (n + 1)) * phi[k][0] * m[k] for k in range(25)])  # sum([(mu[k] ** count) * true_phi[k] * data_val[count] * v[k] for k in range(25)])

        x_data = x_data.detach().numpy()
        E_recon = (inv_N * sum([abs(x_data[n][0] - x_tilde[n][0]) ** 2 + abs(x_data[n][1] - x_tilde[n][1]) ** 2
                                for n in range(width - 1)])) ** 0.5
        print("E_recon", E_recon)

        count += 1


"""学習済みのnetを使って，E_eigfuncを計算"""
I_number = 1000
data = np.loadtxt('./data/E_eigfunc_confirm.csv', delimiter=',', dtype=np.float64)
data = torch.tensor(data, dtype=torch.float32)
width = 2
phi_list = [[0 for count in range(I_number)] for j in range(25)]
y_phi_list = [[0 for count in range(I_number)] for j in range(25)]

for count in range(I_number):
    x_data = data[count * width:count * width + width]
    pred_sai = net(x_data)  # count * 50 : count * 50 + 50

    for j in range(25):
        if j < 22:
            phi_list[j][count] = pred_sai[0][j].detach().numpy()
            y_phi_list[j][count] = pred_sai[1][j].detach().numpy()
        elif j == 22:
            phi_list[j][count] = x_data[0][0].detach().numpy()
            y_phi_list[j][count] = x_data[1][0].detach().numpy()
        elif j == 23:
            phi_list[j][count] = x_data[0][1].detach().numpy()
            y_phi_list[j][count] = x_data[1][1].detach().numpy()
        elif j == 24:
            phi_list[j][count] = 0.1
            y_phi_list[j][count] = 0.1


phi_list = phi_list
y_phi_list = y_phi_list
"""E_eigfunc_jを計算"""
E_eigfunc = [0] * 25
for j in range(25):
    E_eigfunc[j] = np.sqrt(1 / I_number * sum([abs(y_phi_list[j][count] - mu[j] * phi_list[j][count]) ** 2
                                   for count in range(I_number)]))
    print("E_eigfunc", E_eigfunc[j])