# import cvxpy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import japanize_matplotlib
# from scipy.integrate import odeint

# import cvxopt
# from cvxopt import matrix
del matplotlib.font_manager.weight_dict['roman']
matplotlib.font_manager._rebuild()
import japanize_matplotlib
import scipy.linalg
import scipy.signal
from l1sample import mpc_modeling_tool_v5
import random
import csv
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import time
from operator import mul
from network import netR as Net

# 回帰NNの学習
class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        # csvデータの読み出し
        dataframe = []
        with open('./dataset/dataset_R.csv', 'r') as f:
            for line in f:
                row = line.strip().split(',')
                dataframe.append(row)
        self.dataframe = dataframe
    # データのサイズ
    def __len__(self):
        return len(self.dataframe)
    # データとラベルの取得
    def __getitem__(self, idx):
        #dataframeから画像へのパスとラベルを読み出す
        t = transforms.ToTensor()
        # labelはint型に変換
        tmp=[]
        for i in range(2, 62, 1):
            tmp.append(float(self.dataframe[idx][i]))
        label = torch.Tensor(tmp)
        # dataはtorch.Tensorに変換しておく必要あり
        # ※画像の場合などは、transformにtransforms.ToTensorを指定して変換
        tmp2=[]
        for i in range(0, 2, 1):
            tmp2.append(float(self.dataframe[idx][i]))
        data = torch.Tensor(tmp2)
        # data, labelの順でリターン
        return data, label

dataset = MyDataset()
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)

# net = nn.Sequential(
#     nn.Linear(2,64),
#     nn.ReLU(),
#     nn.Linear(64,64),
#     nn.ReLU(),
#     nn.Linear(64,64),
#     nn.ReLU(),
#     nn.Linear(64,20)
#     )
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

optimizer = optim.Adam(model.parameters())
fn_loss=nn.MSELoss()
losses=[]
running_loss=0.0
cntfile=[]
cnt=0
epoches=10
batch_size=100

for epoch in range(epoches):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for xx, yy in dataloader:
        cnt+=1

        optimizer.zero_grad() #勾配を初期化
        y_pred = model(xx) #NNに入力して推論

        loss = fn_loss(y_pred.view_as(yy), yy) #損失関数導出,勾配導出
        yy=yy.double()
        y_pred=y_pred.double()
        loss.backward() #逆伝播
        optimizer.step() #勾配更新
        running_loss=loss.item()
        losses.append(running_loss)
        print(running_loss, cnt*batch_size)

torch.save(model.state_dict(), "./params/model_R.pth")

flg, ax = plt.subplots(1)
learn= np.linspace(0, epoches, cnt)
plt.plot(learn, losses, 'r')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid(True)

plt.show()


#学習終わり,推論はじめ
print("---------------------")
A = np.array([[0, -1],
                [2, 0]])
B = np.array([[2], [0]])

C = np.array([[1], [0]])
(nx, nu) = B.shape
Q = np.eye(nx)
R = np.eye(nu)

Ts = 0.05
Th = 3
N=int(Th/Ts)
# a=1 #L1差分ノルム
# b=0.5 #L1ノルム
# c=b #L2ノルム

Ad, Bd, _, _, dt = scipy.signal.cont2discrete((A,B,C,0),Ts)

# # print(Ad)
# # print(Bd)

x0 = np.array([[1.28], [-1.89]]) # init state

simutime=1/Th
itr=int(simutime*Th)
rx0 =[]
rx1 =[]
ru=[]

noisedf = pd.read_csv('./dataset/noise.csv')
noiseLi = noisedf.values.tolist()

xcurr = x0
ucurr = 0

# t1 = time.time()
# l2norm = 0
# l1norm = 0
for count in range(itr):

    # 分類
    data = [xcurr[0,0], xcurr[1,0]]

    # 回帰
    data = torch.tensor(data, dtype=torch.float32)
    output = model(data)
    output = output.detach().tolist()

    ucurr = output
    for horizon in range(N):
        ru.append(ucurr[horizon])
        rx0.append(xcurr[0])
        rx1.append(xcurr[1])

        #外乱生成
        noise = 0
        # noise = noiseLi[count*N+horizon][0]

        xnext = Ad@xcurr + Bd*(ucurr[horizon]+noise) # x(i) → x(i+1)

        xcurr = xnext

flg, ax = plt.subplots(1)
time= np.arange(0, simutime, Ts)
# print("--------")
plt.tick_params(direction='in')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 10
plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ

# plt.plot(time, rx0, label="$x_1$")
# plt.plot(time, rx1, label="$x_2$")

plt.plot(ru, label="$u$")
# plt.plot(time, noiseLi, label="noise")
plt.xlabel("$t$", fontsize=12)
plt.ylabel("$x$/$u$", fontsize=12)
plt.ylim(-2.0, 2.0)
plt.legend()
plt.grid(True)

plt.show()


