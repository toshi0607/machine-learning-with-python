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

#np.matrix(theta).Tが公式…？
def hypothesis(theta, x):
  return sigmoid(np.matrix(theta) * np.matrix(x))

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def cost(x, y , theta):
  h = hypothesis(theta, x)
  return -m / (y.T * np.log(h) + (1 - y).T * np.log(1 - h))

# np.matrix(x).T*(h - y)が公式…？
def dJ_dtj(x, y, theta):
  h = hypothesis(theta, x)
  return np.matrix(x) * (h - y).T

def gradient_descent(x, y, theta):
  costs = np.zeros(num_iters)
  for i in range (num_iters):
    theta = theta - alpha * dJ_dtj(x, y, theta) #実行するとthetaが初期化時2*1だったのが2*2になって落ちる
    costs[i] = cost(x, y, theta)
  return theta, costs

num_iters = 1000
alpha = 0.001

theta = np.zeros((2,1))
theta, cost = gradient_descent(x[:, 1], y, theta)

theta
