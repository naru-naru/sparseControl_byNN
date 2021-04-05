# import cvxpy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import japanize_matplotlib
# from scipy.integrate import odeint

# import cvxopt
# from cvxopt import matrix
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

# モデルの定義


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
# print(len(dataset)) #36300 =11*11*300

# optimizer = optim.Adam(net.parameters())
# fn_loss=nn.BCEWithLogitsLoss()
# losses=[]
# running_loss=0.0
# cntfile=[]
# cnt=0
# epoches=10
# batch_size=100

model.load_state_dict(torch.load("./params/model_R.pth", map_location=device))

print("----------------------------")

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
# c=1 #L2ノルム

Ad, Bd, _, _, dt = scipy.signal.cont2discrete((A,B,C,0),Ts)

# # print(Ad)
# # print(Bd)

x0 = np.array([[1.28], [-1.89]]) # init state

simutime=1
# simutime=5
itr=int(simutime*Th)
rx0 =[]
rx1 =[]
ru=[]
noiseLi=[]

xcurr = x0
ucurr = 0

t1 = time.time()
l2norm = 0
l1norm = 0

# noisedf = pd.read_csv('./dataset/noise.csv')
# noiseLi = noisedf.values.tolist()

for count in range(itr):

    # 分類
    data = [xcurr[0,0], xcurr[1,0]]
    data = torch.tensor(data, dtype=torch.float32)
    output = model(data)
    # output = torch.sigmoid(output)
    output = output.detach().tolist()


    # Signs = [] #分類のみ用
    # for fm in range(0, 60, 3):
    #     cnt = [output[fm], output[fm+1], output[fm+2]]

    #     sign = cnt.index(max(cnt)) - 1
    #     Signs.append(sign) #分類のみ用

    # 分類のみの場合
    ucurr = output
    for horizon in range(N):
        # 閾値
        if ucurr[horizon] < 0.1 and ucurr[horizon] > -0.1:
            ucurr[horizon] = 0

        ru.append(ucurr[horizon])
        rx0.append(xcurr[0])
        rx1.append(xcurr[1])
        l1norm += abs(ucurr[horizon])
        l2norm += (xcurr[0])**2

        #外乱を与える
        noise = 0
        # noise = noiseLi[count*N+horizon][0]
        # xcurr[0] += 0.5*noise

        xnext = Ad@xcurr + Bd*(ucurr[horizon]+noise) # x(i) → x(i+1)

        xcurr = xnext

t2 = time.time()


fig, ax = plt.subplots(1)
# time= np.arange(0, simutime, Ts)

# 目盛を内側に
plt.tick_params(direction='in')
# plt.ylim(-1.1, 1.1)


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 10
plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ


plt.plot(ru, ls="-", color="g", linewidth="2")
plt.plot(rx0, ls="-", color="b", linewidth="2")
plt.plot(rx1, ls="-", color="r", linewidth="2")
plt.grid(True)
\

# ラベル　

plt.xlabel("$k$", fontsize=24)
plt.ylabel("$u$", fontsize=24)




# fig, ax = plt.subplots(2, 1)
# time= np.arange(0, simutime, Ts)

# # 目盛を内側に
# ax[0].tick_params(direction='in')
# ax[1].tick_params(direction='in')


# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["mathtext.fontset"] = "stix"
# plt.rcParams["font.size"] = 10
# plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ


# ax[0].plot(rx0, ls="-", color="b")
# ax[1].plot(ru, ls="-", color="g")

# # ax[1].plot(time, rx1, ls="-", label="Classification")

# # ax[2].plot(time, ru, ls="-", label="Classification")

# # plt.plot(time, noiseLi, label="noise")
# # plt.ylim(-5.0, 5.0)

# # 凡例・グリッド
# # plt.legend(loc='upper right', borderaxespad=0)
# ax[0].grid(True)
# ax[1].grid(True)
# # ax[0].ylim(-2.0, 2.0)
# # ax[1].ylim(-2.0, 2.0)
# # ax[1].legend(loc='upper right', borderaxespad=0)
# # ax[1].grid(True)
# # ax[2].legend(loc='lower right', borderaxespad=0)
# # ax[2].grid(True)

# # ラベル　

# ax[0].set_xlabel("$t$", fontsize=18)
# ax[0].set_ylabel("$x$", fontsize=18)
# ax[1].set_xlabel("$t$", fontsize=18)
# ax[1].set_ylabel("$u$", fontsize=18)


fig.tight_layout()
plt.show()

stop = [i for i in range(len(ru)) if abs(ru[i])<=0.01]
stopRate = 100 * (len(stop)/len(ru))
print(stopRate)
print(l2norm)
print(t2 - t1)

