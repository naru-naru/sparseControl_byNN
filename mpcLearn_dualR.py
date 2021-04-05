# import cvxpy
import numpy as np
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
from network import netC as Net1
from network import netR as Net2


# # 分類NNの学習
# class MyDataset(torch.utils.data.Dataset):
#     def __init__(self):
#         # csvデータの読み出し
#         dataframe = []
#         with open('./dataset/dataset_C.csv', 'r') as f:
#             for line in f:
#                 row = line.strip().split(',')
#                 dataframe.append(row)
#         self.dataframe = dataframe
#     # データのサイズ
#     def __len__(self):
#         return len(self.dataframe)
#     # データとラベルの取得
#     def __getitem__(self, idx):
#         #dataframeから画像へのパスとラベルを読み出す
#         t = transforms.ToTensor()
#         # labelはint型に変換
#         tmp=[]
#         for i in range(2, 62, 1):
#             tmp.append(float(self.dataframe[idx][i]))
#         label = torch.Tensor(tmp)
#         # dataはtorch.Tensorに変換しておく必要あり
#         # ※画像の場合などは、transformにtransforms.ToTensorを指定して変換
#         data = torch.Tensor([float(self.dataframe[idx][0]), float(self.dataframe[idx][1])])
#         # data, labelの順でリターン
#         return data, label

# dataset = MyDataset()
# # dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)

# # net = nn.Sequential(
# #     nn.Linear(2,64),
# #     nn.ReLU(),
# #     nn.Linear(64,64),
# #     nn.ReLU(),
# #     nn.Linear(64,64),
# #     nn.ReLU(),
# #     nn.Linear(64,60)
# #     )
# # print(len(dataset)) #36300 =11*11*300
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model1 = Net1().to(device)

# optimizer = optim.Adam(model1.parameters())
# fn_loss=nn.BCEWithLogitsLoss()
# losses=[]
# running_loss=0.0
# cntfile=[]
# cnt=0
# epoches=10
# batch_size=100

# for epoch in range(epoches):
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     for xx, yy in dataloader:
#         cnt+=1

#         optimizer.zero_grad() #勾配を初期化
#         y_pred = model1(xx) #NNに入力して推論

#         loss = fn_loss(y_pred.view_as(yy), yy) #損失関数導出,勾配導出
#         yy=yy.double()
#         y_pred=y_pred.double()
#         loss.backward() #逆伝播
#         optimizer.step() #勾配更新
#         running_loss=loss.item()
#         losses.append(running_loss)
#         print(running_loss, cnt*batch_size)

# torch.save(model1.state_dict(), "./params/model_dualC.pth")


# flg, ax = plt.subplots(1)
# learn= np.linspace(0, epoches, cnt)
# plt.plot(learn, losses, 'r')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid(True)

# plt.show()

print("----------------------------")
# 回帰NNの学習
class MyDataset2(torch.utils.data.Dataset):
    def __init__(self):
        # csvデータの読み出し
        dataframe = []
        with open('./dataset/dataset_dualR.csv', 'r') as f:
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

dataset2 = MyDataset2()
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)

# net2 = nn.Sequential(
#     nn.Linear(22,64),
#     nn.ReLU(),
#     nn.Linear(64,64),
#     nn.ReLU(),
#     nn.Linear(64,64),
#     nn.ReLU(),
#     nn.Linear(64,20)
#     )
device2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model2 = Net2().to(device2)

optimizer2 = optim.Adam(model2.parameters())
fn_loss=nn.MSELoss()
losses2=[]
running_loss2=0.0
cntfile2=[]
cnt=0
epoches=10
batch_size=100

for epoch in range(epoches):
    dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=batch_size, shuffle=True)
    for xx, yy in dataloader2:
        cnt+=1

        optimizer2.zero_grad() #勾配を初期化
        y_pred2 = model2(xx) #NNに入力して推論

        loss2 = fn_loss(y_pred2.view_as(yy), yy) #損失関数導出,勾配導出
        yy=yy.double()
        y_pred2=y_pred2.double()
        loss2.backward() #逆伝播
        optimizer2.step() #勾配更新
        running_loss2=loss2.item()
        losses2.append(running_loss2)
        print(running_loss2, cnt*batch_size)

torch.save(model2.state_dict(), "./params/model_dualR.pth")

flg, ax = plt.subplots(1)
learn= np.linspace(0, epoches, cnt)
plt.plot(learn, losses2, 'r')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid(True)

plt.show()


#学習終わり,推論はじめ
# print("---------------------")
# A = np.array([[0, -1],
#                 [2, 0]])
# B = np.array([[2], [0]])

# C = np.array([[1], [0]])
# (nx, nu) = B.shape
# Q = np.eye(nx)
# R = np.eye(nu)

# Ts = 0.05
# Th = 1
# N=int(Th/Ts)
# # a=1 #L1差分ノルム
# # b=0.5 #L1ノルム
# # c=1 #L2ノルム

# Ad, Bd, _, _, dt = scipy.signal.cont2discrete((A,B,C,0),Ts)

# # # print(Ad)
# # # print(Bd)

# x0 = np.array([[1.28], [1.89]]) # init state

# simutime=20
# itr=int(simutime*Th)
# rx0 =[]
# rx1 =[]
# ru=[]
# noiseLi=[]

# xcurr = x0
# ucurr = 0

# output1 = []
# output2 = []
# # t1 = time.time()
# # l2norm = 0
# # l1norm = 0

# noisedf = pd.read_csv('./dataset/noise.csv')
# noiseLi = noisedf.values.tolist()

# for count in range(itr):

#     # 分類
#     data = [xcurr[0,0], xcurr[1,0]]
#     data = torch.tensor(data, dtype=torch.float32)
#     output = model1(data)
#     # output = torch.sigmoid(output)
#     output = output.detach().tolist()

#     data2 = [xcurr[0,0], xcurr[1,0]]

#     Signs = []
#     for fm in range(0, 60, 3):
#         cnt = [output[fm], output[fm+1], output[fm+2]]

#         sign = cnt.index(max(cnt)) - 1
#         data2.append(sign) #カスケード
#         Signs.append(sign)

#     # 回帰
#     data2 = torch.tensor(data2, dtype=torch.float32)
#     output2 = model2(data2)
#     output2 = output2.detach().tolist()

#     ucurr = list(map(mul, output2, Signs))


#     # u入力
#     for horizon in range(N):
#         ru.append(ucurr[horizon])
#         rx0.append(xcurr[0])
#         rx1.append(xcurr[1])

#         #外乱生成
#         noise = 0
#         # noise = noiseLi[count*N+horizon][0]

#         # rate = (np.random.rand() <= 0.1)
#         # if rate:
#         #     noise = np.random.normal(0, 1)
#         # noiseLi.append(noise)

#         xnext = Ad@xcurr + Bd*(ucurr[horizon]+noise) # x(i) → x(i+1)

#         xcurr = xnext

# flg, ax = plt.subplots(1)
# time= np.arange(0, simutime, Ts)
# # print("--------")
# plt.plot(time, rx0, label="x0")
# plt.plot(time, rx1, label="x1")

# plt.plot(time, ru, label="u")
# # plt.plot(time, noiseLi, label="noise")
# plt.ylim(-3.0, 3.0)
# plt.legend()
# plt.grid(True)

# plt.show()


