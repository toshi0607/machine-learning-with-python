import numpy as np
import matplotlib.pyplot as plt

m = 50
theta0 = 0.5
theta1 = 0.2
alpha = 0.01
iterations = 1000

x = np.random.rand(m) * 20 + 20
e = np.random.normal(0, 0.5, m)
y = theta0 + theta1 * x + e
plt.scatter(x,y)


theta = gradientDescent(x, y, alpha, iterations)

plt.show()

def computeCost0(x, y, theta, iterations):
    sum = 0
    for i in range(iterations-1):
         sum += (2.0 / m) * (theta[0] - theta[1] * x[i] - y[i])**2
    return sum

def computeCost1(x, y, theta, iterations):
    sum = 0
    for i in range(iterations-1):
        sum += (2.0 / m) * ((theta[0] - theta[1] * x[i] - y[i]) * x[i])**2
    return sum

def gradientDescent(x, y, alpha, iterations):
    m = len(y)
    #J_history = []
    theta = [0,0]
    for i in range(iterations-1):
        temp0 = theta[0] - alpha * computeCost0(x, y, theta, m)
        temp1 = theta[1] - alpha * computeCost1(x, y, theta, m)
        theta[0] = temp0
        theta[1] = temp1
        #J_history.append(theta)
    return theta
