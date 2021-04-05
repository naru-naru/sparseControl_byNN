# -*- coding: utf-8 -*-
import cvxpy
import numpy as np
import matplotlib.pyplot as plt

import cvxopt
from cvxopt import matrix
import scipy.signal

from l1sample import mpc_modeling_tool_v6, mpc_modeling_tool_v7

import csv
import pandas as pd

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
d = a

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

	# 最適化
	ucurr = mpc_modeling_tool_v6(Ad, Bd, N, P, Q, xcurr, a, b, c, d, umin, umax) #regress
	# ucurr = mpc_modeling_tool_v7(Ad, Bd, N, P, Q, xcurr, a, b, c, d, umin, umax) #dual
	ucurr = ucurr[0]

	dataset.extend(ucurr)

	data = pd.DataFrame([dataset])
	# data.to_csv('./dataset/dataset_dualR.csv', mode="a", index=False, header=False)
	data.to_csv('./dataset/dataset_R.csv', mode="a", index=False, header=False)

	print(count+1)




