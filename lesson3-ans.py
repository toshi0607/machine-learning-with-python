import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
%matplotlib inline

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

def hypothesis(theta, x):
  return sigmoid(np.inner(theta, x))

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def cost(x, y , theta):
  h = hypothesis(theta, x)
  return np.sum(-1/m * (y * np.log(h) + (1 - y) * np.log(1 - h)))

def dJ_dtj(x, y, theta):
  h = hypothesis(theta, x)
  return np.inner(x.T, (h - y))

from scipy.optimize import minimize

print(x)
print(y)
cost2 = lambda t: cost(x, y, t)
print(cost2(np.ones(2)))

res = minimize(cost2, np.array([-8, 1.6]))
print(res)

def gradient_descent(x, y, theta):
  costs = np.zeros(num_iters)
  for i in range (num_iters):
    theta = theta - (alpha * dJ_dtj(x, y, theta))
    costs[i] = cost(x, y, theta)
  return theta, costs

num_iters = 10000
alpha = 0.001

theta = np.array([0, 0])
theta, cost = gradient_descent(x, y, theta)

theta


x1 = np.arange(0, 30, 0.1)
y1 = expit(-9.35488066 + 1.84326689* x1)

plt.xlim(-10, 20)
plt.ylim(-0.1, 1.1)
plt.plot(x[:, 1], y, "ko");
plt.plot(x1, y1)

plt.plot(cost)
plt.xlabel("num_iters")
plt.ylabel("J(theta)")
