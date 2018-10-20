import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + math.e ** (-x))

a = np.array([[1, 2, -3, 0, 1, -3], [3, 1, 2, 1, 0, 2], [2, 2, 2, 2, 2, 1], [1, 0, 2, 1, -2, 2]])
b = np.array([[1, 2, -2, 1], [1, -1, 1, 2], [3, 1, -1, 1]])

x = np.array([1,1,0,0,1,1])
a1 = np.matmul(a, x) + 1
print('a1 = ', a1)
z1 = np.array(list(map(sigmoid, a1)))
print('z1 = ', z1)
b1 = np.matmul(b, z1) + 1
print('b1 = ', b1)
y1 = np.array(list(map(lambda x: math.e ** x, b1)))
y1 /= sum(y1)
print('y1 = ', y1)
y = np.array([0,1,0])
print('loss = ', -math.log(y1[1]))

dJdy2 = 1 / y1[1]      # 1 / (y1[1] * ln2)
dy2db2 = math.exp(b1[1]) * (math.exp(b1[0]) + math.exp(b1[1])) / ((math.exp(b1[0]) + math.exp(b1[1]) + math.exp(b1[2])) ** 2)
db2dbeta21 = z1[0]
dJdbeta21 = dJdy2 * dy2db2 * db2dbeta21
print('gradient of beta21 = ', dJdbeta21)
print('upgraded beta21 = ', b[1][0] - dJdbeta21)
