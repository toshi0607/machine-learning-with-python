# -*- coding: utf-8 -*-
import scipy.io
import scipy.misc
import scipy.optimize
import scipy.special
from numpy import *
from matplotlib import pyplot

digits_data_path = './data/ex3data1.mat'
weights_data_path = './data/ex3weights.mat'

def loadmat(file_path, *names):
    mat = scipy.io.loadmat(file_path)
    # type(mat)
    # -> <class 'dict'>
    return [mat[name] for name in names]

def sigmoid(z):
    return scipy.special.expit(z)

X, y = loadmat(digits_data_path, 'X', 'y')
theta1, theta2 = loadmat(weights_data_path, 'Theta1', 'Theta2')
m, _n = X.shape

# xの数（m * nのm分1をmerge）
X = c_[ones((m, 1)), X]

# 中間層の処理
A = c_[ones((m, 1)), sigmoid(theta1.dot(X.T)).T]

# 出力層の処理
out = theta2.dot(A.T).T
