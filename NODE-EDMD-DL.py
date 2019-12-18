import os
import argparse
import time
import numpy as np
import torchsummary
import modelsummary

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt

from torchdiffeq import odeint


d = 2
l = 150
M = 22  # 22

middle = 50

class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(l, middle),
            nn.Tanh(),
            nn.Linear(middle, l),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                # ここ何をやっている？
                nn.init.normal_(m.weight, mean=0, std=0.1)  # m.weightが変わる．入力テンソルを正規分布から引き出された値で埋めます（mean、std ^ 2）N（平均、標準）
                nn.init.constant_(m.bias, val=0)  # 入力テンソルを値で埋めます val=0

    def forward(self, t, y):
        return self.net(y**3)

before_net = nn.Sequential(
    nn.Linear(d, l),
    nn.Tanh(),
)

after_net = nn.Sequential(
    nn.Linear(l, M),
)

params = {}
params['data_name'] = 'Duffing_oscillator'
data = np.loadtxt(('./data/%s_train_x.csv' % (params['data_name'])), delimiter=',', dtype=np.float64)  # ここでデータを読み込む
data = torch.tensor(data, dtype=torch.float32)


def J(K, theta):
    pass


lambda_ = 1e-3
I = torch.tensor(np.eye(25, 25), dtype=torch.float32)
# K_tilde = np.linalg.pinv(G + lambda_.dot(I)).dot(A)
epsilon = 0.1
net = ODEFunc()
# optimizer = optim.SGD(net.parameters(), lr=1e-5)
optimizer = optim.Adam(net.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()  # J(K, theta)
y = []

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

def total_net(data):
    before_pred_sai = before_net(data)
    after_pred_sai = odeint(net, before_pred_sai, tSpan)[0]
    pred_sai = after_net(after_pred_sai)
    return pred_sai

tSpan = np.arange(1, 1 + 0.1, 5)
tSpan = torch.from_numpy(tSpan)
# intSpan = torch.tensor([i for i in range(1, 26)], dtype=torch.float32)
width = 11  # 11
inv_N = 1/10  # 0.1
min_loss = float("INF")

# パラメータカウント
params = 0
for p in before_net.parameters():
    if p.requires_grad:
        params += p.numel()
for p in net.parameters():
    if p.requires_grad:
        params += p.numel()
for p in after_net.parameters():
    if p.requires_grad:
        params += p.numel()
print("parameterの数", params)
exit()

# while J(K, theta) > epsilon:
for count in range(1000):
    optimizer.zero_grad()

    x_data = data[count * width:count * width + width - 1]
    y_data = data[count * width + 1:count * width + width]
    #t = data_val[count * width:count * width + width]
    """pred_sai = odeint(net, data_val[count * width:count * width + width - 1], tSpan)  # count * 50 : count * 50 + 50
    y_pred_sai = odeint(net, data_val[count * width + 1:count * width + width, :], tSpan)"""

    pred_sai = total_net(x_data)
    y_pred_sai = total_net(y_data)

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
    QWRETY = y_pred_sai_T - Pred  # pred_sai_T
    # torch.matrix_power(QWRETY)
    for i in range(width - 1):
        # print(QWRETY[i])
        # loss += torch.log(sum([abs(c) for c in QWRETY[i]]))  # 順番逆かも，結果は変わらない
        for c in QWRETY[:, i]:
            loss += c ** 2
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
K = K_tilde  # torch.rand(25, 25) #K_tilde
mu = 0
for tr_val_te in ["E_recon_50"]:
    data = np.loadtxt('./data/E_recon_50.csv', delimiter=',', dtype=np.float64)
    data = torch.tensor(data, dtype=torch.float32)
    count = 0
    width = 51
    """Bを計算，X=BΨ"""
    X25 = data[0].view(2, -1)
    for i in range(1, M + 3):
        x2_data = data[width * i].view(2, -1)
        X25 = torch.cat([X25, x2_data], dim=1)

    Sai = total_net(data[0])

    tmp = data[0].detach().tolist() + [0.1]  # [i + [0.1] for i in data[0].detach().tolist()]
    fixed_sai1 = torch.tensor(tmp, dtype=torch.float32)
    Sai = torch.cat([Sai, fixed_sai1])
    Sai = Sai.view(M + 3, -1)
    for i in range(1, M + 3):
        sai = total_net(data[width * i])
        tmp = data[width * i].detach().tolist() + [0.1]
        fixed_sai = torch.tensor(tmp, dtype=torch.float32)
        sai = torch.cat([sai, fixed_sai])
        sai = sai.view(M + 3, -1)
        Sai = torch.cat([Sai, sai], dim=1)

    # Sai = torch.transpose(Sai, 0, 1)
    B = torch.mm(X25, torch.inverse(Sai))
    B = B.detach().numpy()
    K = K.detach().numpy()
    while count < 1000:
        width = 51
        x_data = data[count * width:count * width + width - 1]  # N = 10

        sai = total_net(x_data)

        fixed_sai = torch.tensor([i + [0.1] for i in x_data.detach().tolist()], dtype=torch.float32)
        sai = torch.cat([sai, fixed_sai], dim=1).detach().numpy()
        sai_T = sai.T
        """E_reconを計算"""

        mu, xi, zeta = la.eig(K, left=True, right=True)

        # mu zeta = K zeta
        # mu z = K.T z
        # mu z.T = z.T K，xi = z.T
        mu2, _, z = la.eig(K.T, left=True, right=True)

        # print(mu2[1] * z[:, 1].T - z[:, 1].T.dot(K))

        print(mu[1] * zeta[:, 1] - K.dot(zeta[:, 1]))
        print(mu[1] * xi[:, 1].T - xi[:, 1].T.dot(K))

        mu_real = [i.real for i in mu]
        mu_imag = [i.imag for i in mu]
        plt.scatter(mu_real, mu_imag)
        plt.show()

        m = B.dot(zeta)  # (xi.T.dot(B)).T  # 本当はエルミート
        m = m.T
        phi = (xi.T).dot(sai_T)

        x_tilde = [[0, 0] for _ in range(width - 1)]  # [[0, 0]] * (width - 1)
        x_tilde_phi = [[0, 0] for _ in range(width - 1)]
        x_tilde[0][0] = x_data[0][0]
        x_tilde[0][1] = x_data[0][1]
        x_tilde_phi[0][0] = x_data[0][0]
        x_tilde_phi[0][1] = x_data[0][1]
        for n in range(1, width - 1):
            x_tilde[n][0] = sum([(mu[k] ** n) * phi[k][0] * m[k][0] for k in range(
                M + 3)]).real  # sum([(mu[k] ** count) * true_phi[k] * data_val[count] * v[k] for k in range(25)])
            x_tilde[n][1] = sum([(mu[k] ** n) * phi[k][0] * m[k][1] for k in range(M + 3)]).real

            x_tilde_phi[n][0] = sum([phi[k][n] * m[k][0] for k in range(
                M + 3)]).real  # sum([(mu[k] ** count) * true_phi[k] * data_val[count] * v[k] for k in range(25)])
            x_tilde_phi[n][1] = sum([phi[k][n] * m[k][1] for k in range(M + 3)]).real

        x_data = x_data.detach().numpy()
        E_recon = (inv_N * sum([abs(x_data[n][0] - x_tilde[n][0]) ** 2 + abs(x_data[n][1] - x_tilde[n][1]) ** 2
                                for n in range(width - 1)])) ** 0.5
        print("E_recon", E_recon)

        count += 1
        plt.plot(x_data[:, 0], label="correct")  # 実データ，青
        x_tilde_0 = [i for i, j in x_tilde]
        plt.plot(x_tilde_0, label="predict")  # 予測，オレンジ
        x_tilde_phi_0 = [i for i, j in x_tilde_phi]
        plt.plot(x_tilde_phi_0, label="phi_pred")  # 予測Φ，緑
        plt.legend()
        plt.show()

"""学習済みのnetを使って，E_eigfuncを計算"""
I_number = 1000
data = np.loadtxt('./data/E_eigfunc_confirm.csv', delimiter=',', dtype=np.float64)
data = torch.tensor(data, dtype=torch.float32)
width = 2
phi_list = [[0 for count in range(I_number)] for j in range(25)]
y_phi_list = [[0 for count in range(I_number)] for j in range(25)]

for count in range(I_number):
    x_data = data[count * width:count * width + width]
    pred_sai = total_net(x_data)

    for j in range(M + 3):
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
            y_phi_list[j][count] = 0.1

phi_list = phi_list
y_phi_list = y_phi_list
"""E_eigfunc_jを計算"""
E_eigfunc = [0] * (M + 3)
for j in range(M + 3):
    E_eigfunc[j] = np.sqrt(1 / I_number * sum([abs(y_phi_list[j][count] - mu[j] * phi_list[j][count]) ** 2
                                               for count in range(I_number)]))
    print("E_eigfunc", E_eigfunc[j])
