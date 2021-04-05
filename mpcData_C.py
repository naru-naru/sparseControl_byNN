# -*- coding: utf-8 -*-
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
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import Dataset
# from sklearn.model_selection import train_test_split
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader, Dataset
# import time
# from operator import mul

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

# print(N)
# N=10
Ad, Bd, _, _, dt = scipy.signal.cont2discrete((A,B,C,0),Ts)

a=10 #L1差分ノルム
b=0.5 #L2ノルム
c=b #L2終端ノルム
d = 0

umax= 1
umin= -umax
itr = 20000

# 予測ホライズン内の[状態：制御入力]を学習
for count in range(itr):
	dataset = [] #データ1個分

	# 初期状態
	x1 = 4 * (np.random.rand() - 0.5)
	x2 = 4 * (np.random.rand() - 0.5)
	dataset.append(x1) #入力データ
	dataset.append(x2) #入力データ


	xcurr = np.array([[x1], [x2]])
	ucurr = mpc_modeling_tool_v6(Ad, Bd, N, P, Q, xcurr, a, b, c, d, umin, umax)
	ucurr = ucurr[0]

	for i in range(len(ucurr)): # K ~ K+n-1 20ステップ分

		cnt = [0] * 2
		active = 1
		if ucurr[i] > 0.1:
			active += 1
		elif ucurr[i] < -0.1:
			active -= 1
		cnt.insert(active, 1) #分類用ラベル作成

		dataset.extend(cnt) #分類用

	# 各ステップごとに最適入力データ格納
	dataset.extend(ucurr)

	data = pd.DataFrame([dataset])
	data.to_csv('./dataset/dataset_C.csv', mode="a", index=False, header=False) #dual
	# data.to_csv('./dataset/dataset_C.csv', mode="a", index=False, header=False) #class

	print(count+1)




