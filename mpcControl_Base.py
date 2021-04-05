# import cvxpy
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
# from scipy.integrate import odeint

# import cvxopt
# from cvxopt import matrix
import scipy.linalg
import scipy.signal
from l1sample import mpc_modeling_tool_v6
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


A = np.array([[0, -1],
                [2, 0]])
B = np.array([[2], [0]])

C = np.array([[1], [0]])
(nx, nu) = B.shape
Q = np.eye(nx)
P = np.eye(nx)

Ts = 0.05
Th = 3
N=int(Th/Ts)
a=10 #L1ノルム
b=0.5 #L2ノルム
c=b #L2終端ノルム
d = 0

Ad, Bd, _, _, dt = scipy.signal.cont2discrete((A,B,C,0),Ts)

umax= 1
umin= -umax

x0 = np.array([[1.28], [-1.89]]) # init state

simutime=10/3
simutime=1
itr=int(simutime*Th)
rx0 =[]
rx1 =[]
ru=[]
noiseLi=[]

xcurr = x0
ucurr = 0

output1 = []
output2 = []

l2norm = 0
l1norm = 0

noisedf = pd.read_csv('./dataset/noise.csv')
noiseLi = noisedf.values.tolist()

t1 = time.time()
for count in range(itr):

    ucurr = mpc_modeling_tool_v6(Ad, Bd, N, P, Q, xcurr, a, b, c, d, umin, umax)
    ucurr = ucurr[0]

    for horizon in range(N):
        ru.append(ucurr[horizon])
        rx0.append(xcurr[0][0])
        rx1.append(xcurr[1][0])
        l1norm += abs(ucurr[horizon])
        l2norm += (xcurr[0])**2

        #外乱を与える
        noise = 0
        # noise = noiseLi[count*N+horizon][0]

        xnext = Ad@xcurr + Bd*(ucurr[horizon]+noise) # x(i) → x(i+1)

        xcurr = xnext

t2 = time.time()

fig, ax = plt.subplots(1)
# time= np.arange(0, simutime, Ts)

# 目盛を内側に
plt.tick_params(direction='in')

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 10
plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ


# plt.plot(time, ru, ls="-")
plt.plot(ru, ls="-", color="b", linewidth="2")


# ax[1].plot(time, rx1, ls="-", label="Classification")

# ax[2].plot(time, ru, ls="-", label="Classification")

# plt.plot(time, noiseLi, label="noise")
# plt.ylim(-5.0, 5.0)

# 凡例・グリッド
# plt.legend(loc='upper right', borderaxespad=0)
plt.grid(True)
plt.ylim(-0.05, 0.05)
plt.xlim(20, )
# ax[1].legend(loc='upper right', borderaxespad=0)
# ax[1].grid(True)
# ax[2].legend(loc='lower right', borderaxespad=0)
# ax[2].grid(True)

# ラベル　
plt.xlabel("$k$", fontsize=24)
plt.ylabel("$u$", fontsize=24)
# ax[1].set_xlabel("$t$", fontsize=18)
# ax[1].set_ylabel("$x_2$", fontsize=18)
# ax[2].set_xlabel("$t$", fontsize=18)
# ax[2].set_ylabel("$u$", fontsize=18)


fig.tight_layout()
plt.show()

stop = [i for i in range(len(ru)) if abs(ru[i])<=0.01]
stopRate = 100 * (len(stop)/len(ru))
# print(stopRate)
# print(l2norm[0])
# print(t2 - t1)

# print(rx0)

