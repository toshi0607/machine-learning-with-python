import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

m = 30
n = 1
theta0, theta1 = -8, 1.6

x0 = np.arange(0, 30, 0.1)
y0 = expit(theta0 + theta1 * x0)

x = np.ones((m, n+1))
x[:, 1] = np.random.rand(m) * 10
y = np.ceil(expit(theta0 + theta1 * x[:, 1]) - np.random.rand(m))

plt.xlim(0, 12)
plt.ylim(-0.1, 1.1)
plt.title("Tumors")
plt.xlabel("size (mm)")
plt.ylabel("malignant?")

plt.plot(x0, y0)
plt.plot(x[:, 1], y, "ko");

def hypothesis(theta):
  return lambda x: sigmoid(np.matrix(theta).T * np.matrix(x))

def sigmoid():
  return lambda  z: 1 / (1 + exp(-z))

def cost(X, y , theta):
  h = hypothesis(theta)
  return -1 * m / (y.T * np.log(h(X)) + (1 - y).T * np.log(1 - h(X)))

def dJ_dtj(X, y, theta):
  h = hypothesis(theta)
  return np.matrix(X).T * (h(X) - y)

def gradient_descent(X, y, theta):
  costs = np.zeros(num_iters)

  for i in range (num_iters):
    theta = theta - alpha * dJ_dtj(X, y, theta)
    costs[i] = cost(X, y, theta)
  return theta, costs
