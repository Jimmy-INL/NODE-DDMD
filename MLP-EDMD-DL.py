
import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import cloudpickle

import numpy as np
from scipy import linalg as la

from matplotlib import pyplot as plt
from scipy.stats import uniform
from statistics import mean


data_name = 'spectrum'  # 'Duffing_oscillator', 'Linear'，Discrete_Linear Duffing_oscillator spectrum-1



def J(K, theta):
    pass



lambda_ = 1e-2  # 1e-6

# K_tilde = np.linalg.pinv(G + lambda_.dot(I)).dot(A)
epsilon = 20

d = 2
l = 100  # 70
M = 22  # 22
I = torch.eye(M + 3, M + 3)

N = 10000
#width = 11  #11
inv_N = 1/N  #0.1

net = nn.Sequential(
    nn.Linear(d, l),
    nn.Tanh(),
    #nn.ReLU(),
    nn.Linear(l, l),
    nn.Tanh(),
    #nn.ReLU(),
    #nn.Dropout(0.5),
    nn.Linear(l, l),
    nn.Tanh(),
    #nn.ReLU(),
    nn.Linear(l, M),
)


with open('MLP_net.pkl', 'rb') as f:  # nolta_
    net = cloudpickle.load(f)

# optimizer = optim.SGD(net.parameters(), lr=2e-4)
optimizer = optim.Adam(net.parameters(), lr=1e-3)  # 1e-4
loss_fn = nn.MSELoss()  # J(K, theta)

def data_Preprocessing(tr_val_te, cut):
    data = np.loadtxt(('./data/%s_%s.csv' % (data_name, tr_val_te)), delimiter=',', dtype=np.float64)[:cut]
    # np.loadtxt(('./data/%s_val_x.csv' % (params['data_name'])), delimiter=',', dtype=np.float64)  # ここでデータを読み込む
    data = torch.tensor(data, dtype=torch.float32)
    return data

def p_inv(X):
    X_T = torch.transpose(X, 0, 1)
    return torch.mm(torch.inverse(torch.mm(X_T, X)), X_T)  # (X_TX)-1X_T

def Frobenius_norm(X):
    M = torch.mm(X, torch.transpose(X, 0, 1))
    return torch.sum(torch.diag(M, 0))

"""def graph__________(y, st):
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
    plt.show()"""

plt.rcParams["font.size"] = 18
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
#plt.rcParams['text.usetex'] = True
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
        plt.plot(correct, label="exact")  # 実データ，青
        #plt.plot(predict, label="predict")  # 予測，オレンジ
        plt.scatter([i for i in range(50)], predict, label="predicted", color="orange")  # 予測，オレンジ
        #plt.plot(phi_predict, label="predict")  # 予測Φ，緑
        plt.title(name[:2] + "_trajectory")
        plt.xlabel('n')
        plt.ylabel(name[:2])
        plt.legend()

        plt.tight_layout()
    plt.savefig("MLP_img/png/" + name + ".png")
    plt.savefig("MLP_img/eps/" + name + ".eps")
    #plt.show()



# while J(K, theta) > epsilon:
x = []
y = []
X = []
Y = []
count = 0
K_tilde = []

"""netを学習"""
x_data = data_Preprocessing("train_x", N)
y_data = data_Preprocessing("train_y", N)
"""data = np.loadtxt('./data/E_recon_50.csv', delimiter=',', dtype=np.float64)
data = torch.tensor(data, dtype=torch.float32)"""
# if tr_val_te != "train":


#exit()

count = 0
rotation = 10000


#lambda_ = 1e-2 l = 100 lr=1e-3 rotation = 1500
# パラメータカウント
params = 0
for p in net.parameters():
    if p.requires_grad:
        params += p.numel()
print("parameterの数", params)
# summary(net, (2, 10000, ))
# modelsummary.summary(net, d, M)
#exit()"""

loss = float("INF")
while loss > epsilon:  # count < rotation and
    if count % 100 == 0:
        print(count)
    optimizer.zero_grad()

    #x_data = data[count * width:count * width + width - 1]  # 0～9，11～20，

    #y_data = data[count * width + 1:count * width + width]  # 1～10，12～21，
    pred_sai = net(x_data)  # count * 50 : count * 50 + 50
    y_pred_sai = net(y_data)

    #fixed_sai = torch.cat([x_data, torch.tensor([[0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1]])], dim=1)
    #y_fixed_sai = torch.cat([y_data, torch.tensor([[0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1]])], dim=1)

    fixed_sai = torch.tensor([i + [0.1] for i in x_data.detach().tolist()], dtype=torch.float32)
    pred_sai = torch.cat([pred_sai, fixed_sai], dim=1)
    y_fixed_sai = torch.tensor([i + [0.1] for i in y_data.detach().tolist()], dtype=torch.float32)
    y_pred_sai = torch.cat([y_pred_sai, y_fixed_sai], dim=1)

    pred_sai_T = torch.transpose(pred_sai, 0, 1)

    G = inv_N * torch.mm(pred_sai_T, pred_sai)  # 本当はエルミート
    A = inv_N * torch.mm(pred_sai_T, y_pred_sai)

    """G = torch.ger(pred_sai[0], pred_sai[0])
    A = torch.ger(pred_sai[0], y_pred_sai[0])
    for i in range(1, N):
        G += torch.ger(pred_sai[i], pred_sai[i])
        A += torch.ger(pred_sai[i], y_pred_sai[i])
    G *= 1 / N
    A *= 1 / N"""

    #K_tilde = torch.mm(p_inv(G + lambda_ * I), A)  # pinverseを使うとおかしくなるのでp_invで代用
    K_tilde = torch.mm(torch.pinverse(G + lambda_ * I), A)
    #K_tilde = torch.mm(torch.inverse(G + lambda_ * I), A)  # + lambda_ * I
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
    PPAP = QWRETY ** 2
    loss += torch.sum(PPAP)
    # torch.matrix_power(QWRETY)
    """for i in range(N):
        # print(QWRETY[i])
        # loss += torch.log(sum([abs(c) for c in QWRETY[i]]))  # 順番逆かも，結果は変わらない
        for c in QWRETY[:, i]:
            loss += c ** 2"""
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

x = [i for i in range(count)]
graph(x, y, "train", "plot")
count = 0

with open('MLP_net.pkl', 'wb') as f:
    cloudpickle.dump(net, f)

"""学習済みのnetを使って，E_reconを計算"""
K = K_tilde # torch.rand(25, 25) #K_tilde
mu = 0
for tr_val_te in ["E_recon_50"]:
    data = data_Preprocessing("train_x", N)
    count = 0
    width = 10
    """Bを計算，X=BΨ"""
    X25 = data[0].view(2, -1)
    for i in range(1, M + 3):
        x2_data = data[width * i].view(2, -1)
        X25 = torch.cat([X25, x2_data], dim=1)

    Sai = net(data[0])
    tmp = data[0].detach().tolist() + [0.1]  # [i + [0.1] for i in data[0].detach().tolist()]
    fixed_sai1 = torch.tensor(tmp, dtype=torch.float32)
    Sai = torch.cat([Sai, fixed_sai1])
    Sai = Sai.view(M + 3, -1)
    for i in range(1, M + 3):
        sai = net(data[width * i])
        tmp = data[width * i].detach().tolist() + [0.1]
        fixed_sai = torch.tensor(tmp, dtype=torch.float32)
        sai = torch.cat([sai, fixed_sai])
        sai = sai.view(M + 3, -1)
        Sai = torch.cat([Sai, sai], dim=1)

    # Sai = torch.transpose(Sai, 0, 1)
    B = torch.mm(X25, torch.inverse(Sai))
    B = torch.tensor([[1 if ((i == 22 and j == 0) or (i == 23 and j == 1)) else 0 for i in range(M + 3)] for j in range(2)])
    B = B.detach().numpy()
    K = K.detach().numpy()

    data = data_Preprocessing("E_recon_50", N)
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
        x_data = data[count * width:count * width + width]  # N = 10
        sai = net(x_data)
        fixed_sai = torch.tensor([i + [0.1] for i in x_data.detach().tolist()], dtype=torch.float32)
        sai = torch.cat([sai, fixed_sai], dim=1).detach().numpy()
        sai_T = sai.T

        """E_reconを計算"""
        #m = B.dot(zeta)  # (xi.T.dot(B)).T  # 本当はエルミート
        #m = m.T
        m = xi.T.dot(B.T)

        # sai_T = torch.rand(M + 3, width - 1) * 100
        #phi = (xi.T).dot(sai_T)
        phi = sai.dot(zeta)
        phi = phi.T

        """for kk in range(M + 3):
            print("-----------------------", kk, "--------------------------------------")
            for i in range(1, width):
                # print(Phi[10][i - 1] / Phi[10][i], Mu[10])
                print(phi[kk][i] / phi[kk][i - 1], mu[kk])"""

        x_tilde = [[0, 0] for _ in range(width)]  # [[0, 0]] * (width - 1)
        x_tilde_phi = [[0, 0] for _ in range(width)]

        x_tilde[0][0] = float(x_data[0][0])
        x_tilde[0][1] = float(x_data[0][1])
        x_tilde_phi[0][0] = float(x_data[0][0])
        x_tilde_phi[0][1] = float(x_data[0][1])
        for n in range(1, width):
            #print((mu[1] ** n) * phi[1][0], phi[1][n])
            x_tilde[n][0] = sum([(mu[k] ** n) * phi[k][0] * m[k][0] for k in range(M + 3)]).real  # sum([(mu[k] ** count) * true_phi[k] * data_val[count] * v[k] for k in range(25)])
            x_tilde[n][1] = sum([(mu[k] ** n) * phi[k][0] * m[k][1] for k in range(M + 3)]).real

            x_tilde_phi[n][0] = sum([phi[k][n] * m[k][0] for k in range(M + 3)]).real  # sum([(mu[k] ** count) * true_phi[k] * data_val[count] * v[k] for k in range(25)])
            x_tilde_phi[n][1] = sum([phi[k][n] * m[k][1] for k in range(M + 3)]).real

        x_data = x_data.detach().numpy()
        E_recon = (inv_N * sum([abs(x_data[n][0] - x_tilde[n][0]) ** 2 + abs(x_data[n][1] - x_tilde[n][1]) ** 2
                                for n in range(width)])) ** 0.5

        count += 1
        print(count, "E_recon", E_recon)

        x_tilde_0 = [i for i, j in x_tilde]
        x_tilde_phi_0 = [i for i, j in x_tilde_phi]

        graph([], [], "x1_traj_" + "{stp:02}".format(stp=count), "multi_plot"
              , x_data[:, 0], x_tilde_0, x_tilde_phi_0)

        x_tilde_1 = [j for i, j in x_tilde]
        x_tilde_phi_1 = [j for i, j in x_tilde_phi]

        graph([], [], "x2_traj_" + "{stp:02}".format(stp=count), "multi_plot"
              , x_data[:, 1], x_tilde_1, x_tilde_phi_1)


"""学習済みのnetを使って，E_eigfuncを計算"""
I_number = 10000
data = data_Preprocessing("E_eigfunc", I_number * width)
width = 2
phi_list = [[0 for count in range(I_number)] for j in range(25)]
y_phi_list = [[0 for count in range(I_number)] for j in range(25)]

for count in range(I_number):
    x_data = data[count * width:count * width + width]
    pred_sai = net(x_data)  # count * 50 : count * 50 + 50
    fixed_sai = torch.tensor([i + [0.1] for i in x_data.detach().tolist()], dtype=torch.float32)
    pred_sai = torch.cat([pred_sai, fixed_sai], dim=1).detach().numpy()
    #pred_phi = (xi.T).dot(pred_sai.T)
    # pred_phi = (xi.T).dot(sai_T)
    pred_phi = pred_sai.dot(zeta)
    pred_phi = pred_phi.T

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
norm = [0] * (M + 3)
for j in range(M + 3):
    norm[j] = [phi_list[j][count].real ** 2 + phi_list[j][count].imag ** 2
            for count in range(I_number)]
    norm[j] = np.sqrt(1 / I_number * sum(norm[j]))

for j in range(M + 3):
    tmp = [(1 / norm[j] * y_phi_list[j][count] - mu[j] * 1 / norm[j] * phi_list[j][count]).real ** 2
           + (1 / norm[j] * y_phi_list[j][count] - mu[j] * 1 / norm[j] * phi_list[j][count]).imag ** 2
                                   for count in range(I_number)]
    E_eigfunc[j] = np.sqrt(1 / I_number * sum(tmp))
    # print(j, tmp)
    print(j, "E_eigfunc", E_eigfunc[j])
print(mean(E_eigfunc))