# print(v)
# pred_sai_2 = np.reshape(pred_sai, (3, 4))
"""true_phi = [0] * 25
pred_sai_np = pred_sai.detach().numpy()
for k in range(25):
    p = xi[k]
    q = pred_sai_np
    true_phi[k] += xi[k].dot(pred_sai_np[0])
#print(true_phi)"""
x1_data = data[0].view(2, -1)
x2_data = data[11].view(2, -1)
X1X2 = torch.cat([x1_data, x2_data], dim=1)
sai1 = net(data[0])  # .view(25, -1)
sai2 = net(data[11])
tmp = data[0].detach().tolist() + [0.1]  # [i + [0.1] for i in data[0].detach().tolist()]
fixed_sai1 = torch.tensor(tmp, dtype=torch.float32)
sai1 = torch.cat([sai1, fixed_sai1])
tmp = data[11].detach().tolist() + [0.1]
fixed_sai2 = torch.tensor(tmp, dtype=torch.float32)
sai2 = torch.cat([sai2, fixed_sai2])

sai1 = sai1.view(25, -1)
sai2 = sai2.view(25, -1)

Sai1Sai2 = torch.transpose(torch.cat([sai1, sai2], dim=1), 0, 1)
B = torch.mm(X1X2, torch.inverse(Sai1Sai2))
B = B.detach().numpy()