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

X, y = loadmat(digits_data_path, 'X', 'y')
theta1, theta2 = loadmat(weights_data_path, 'Theta1', 'Theta2')
