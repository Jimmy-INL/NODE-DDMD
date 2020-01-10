from matplotlib import pyplot as plt

x = [0.1 / 0.7, 0.2 / 0.7, 0.3 / 0.7, 0.4 / 0.7, 0.5 / 0.7,
     0.6 / 0.7, 0.7 / 0.7, 0.8 / 0.7, 0.9 / 0.7, 1.0 / 0.7]

y = [22722 / 10150 for _ in range(10)]

plt.rcParams["font.size"] = 18
name = "efficient_graph"
#plt.plot(correct, label="exact")  # 実データ，青
#plt.plot(predict, label="predict")  # 予測，オレンジ
plt.scatter(x, y)  # 予測，オレンジ
#plt.plot(phi_predict, label="predict")  # 予測Φ，緑
#plt.title()
plt.xlabel("ν/μ")
plt.ylabel("MLP / NODE parameter count")
plt.grid(True)  # 目盛の表示
#plt.legend()
plt.tight_layout()
plt.savefig("png/" + name + ".png")
plt.savefig("eps/" + name + ".eps")
plt.show()