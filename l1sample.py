"""
Model predictive control sample code without modeling tool (cvxpy)
author: Atsushi Sakai
"""

import cvxpy
import numpy as np
import matplotlib.pyplot as plt

import cvxopt
from cvxopt import matrix
import scipy.linalg
# import gurobipy as grb

DEBUG_ = False

print("cvxpy version:", cvxpy.__version__)


def mpc_modeling_tool_v5(A, B, N, P, Q, xcurr, a, b, c, d, umin, umax): #MPC用 入力差分L1ノルム
    """
    solve MPC with modeling tool for test
    """
    (nx, nu) = B.shape

    # mpc calculation
    X = cvxpy.Variable((nx, N + 1))
    U = cvxpy.Variable((nu, N))

    objective = 0.0
    constraints = []


    objective += a * cvxpy.norm(U,1) #L1

    for t in range(N):
        objective += b * cvxpy.quad_form(X[:, t], Q) #L2
    objective += c * cvxpy.quad_form(X[:, N], P) #L2end

    for t in range(N):
        constraints += [X[:, t + 1] == A @ X[:, t] + B @ U[:, t]]

    constraints += [X[:, 0] == xcurr[:, 0]] #初期状態

    constraints += [U[:, :]<=umax, U[:, :]>=umin]

    prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)
    prob.solve(verbose=False)
    if prob.status != 'optimal':
        print(prob.status)
    return U.value

# 通常, #パラレル 分類
def mpc_modeling_tool_v6(A, B, N, P, Q, xcurr, a, b, c, d, umin, umax): #MPC用 入力差分L1ノルム
    """
    solve MPC with modeling tool for test
    """
    (nx, nu) = B.shape

    # mpc calculation
    X = cvxpy.Variable((nx, N + 1))
    U = cvxpy.Variable((nu, N))


    objective = 0.0
    constraints = []

    objective += a * cvxpy.norm(U,1) #L1

    for t in range(N):
        objective += b * cvxpy.quad_form(X[:, t], Q) #L2
    objective += c * cvxpy.quad_form(X[:, N], P) #L2end

    for t in range(N):
        constraints += [X[:, t + 1] == A @ X[:, t] + B @ U[:, t]]

    constraints += [X[:, 0] == xcurr[:, 0]] #初期状態

    constraints += [U[:, 0]<=umax, U[:, 0]>=umin]

    constraints += [U[:, :]<=umax, U[:, :]>=umin]


    prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)
    prob.solve(verbose=False)
    if prob.status != 'optimal':
        print(prob.status)

    return U.value

#パラレル　回帰
def mpc_modeling_tool_v7(A, B, N, P, Q, xcurr, a, b, c, d, umin, umax): #MPC用 入力差分L1ノルム
    """
    solve MPC with modeling tool for test
    """
    (nx, nu) = B.shape

    # mpc calculation
    X = cvxpy.Variable((nx, N + 1))
    U = cvxpy.Variable((nu, N))


    objective = 0.0
    constraints = []

    objective += a * cvxpy.norm(U,1) #L1

    for t in range(N):
        objective += b * cvxpy.quad_form(X[:, t], Q) #L2
    objective += c * cvxpy.quad_form(X[:, N], P) #L2end

    for t in range(N):
        constraints += [X[:, t + 1] == A @ X[:, t] + B @ U[:, t]]

    constraints += [X[:, 0] == xcurr[:, 0]] #初期状態

    constraints += [U[:, 0]<=umax, U[:, 0]>=umin]

    constraints += [U[:, :]<=umax, U[:, :]>=umin]


    prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)
    prob.solve(verbose=False)
    if prob.status != 'optimal':
        print(prob.status)

    for t in range(N):
        U.value[:, t] = abs(U.value[:, t])

    return U.value


