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




class netC(nn.Module):
    def __init__(self):
        super(netC,self).__init__()
        self.linear1 = nn.Linear(2,64)
        self.linear2 = nn.Linear(64,64)
        self.linear3 = nn.Linear(64,180)


    def forward(self,x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)

        return x

class netR(nn.Module):
    def __init__(self):
        super(netR,self).__init__()
        self.linear1 = nn.Linear(2,64)
        self.linear2 = nn.Linear(64,64)
        self.linear3 = nn.Linear(64,60)


    def forward(self,y):
        y = self.linear1(y)
        y = F.relu(y)
        y = self.linear2(y)
        y = F.relu(y)
        y = self.linear2(y)
        y = F.relu(y)
        y = self.linear3(y)

        return y