import numpy as np
import math
import sys


class Sigmoid:
    def sigmoid(self, a):
        return 1 / (1 + math.e ** (-a))

    def forward(self, a):
        b = self.sigmoid(a)
        return b

    def backward(self, a, b, Gb):
        Gb = Gb.reshape((-1))
        Ga = Gb * b * (1 - b)
        # Ga = np.multiply(np.multiply(Gb, b), 1 - b)
        return Ga


class Softmax:
    def softmax(self, a):
        exp_a = math.e ** a
        print(sum(exp_a))
        return exp_a / sum(exp_a)

    def forward(self, a):
        b = self.softmax(a)
        return b

    def backward(self, a, b, Gb):
        res = np.diag(b) - np.dot(b, np.transpose(b))
        Ga = np.dot(np.transpose(Gb), res)
        return Ga


class Linear:
    def forward(self, a, alpha):
        b = np.dot(alpha, a)
        return b

    def backward(self, a, alpha, Gb):
        Gb = Gb.reshape((-1,1))
        a = a.reshape((-1,1))
        if alpha.shape[0] != Gb.shape[0]:
            Gb = Gb[1:]
        G_alpha = np.dot(Gb, np.transpose(a))
        Ga = np.dot(np.transpose(alpha), Gb)
        return G_alpha, Ga


class CrossEntropy:
    def forward(self, a, a_hat):
        b = - np.dot(np.transpose(a), np.log(a_hat))
        return b

    def backward(self, a, a_hat, b, Gb):
        Ga = - Gb * (a / a_hat)
        return Ga


class Net:
    def __init__(self):
        self.linear = Linear()
        self.sigmoid = Sigmoid()
        self.softmax = Softmax()
        self.cross_entropy = CrossEntropy()

    def forward(self, data, label, alpha, beta):
        self.a = self.linear.forward(data, alpha)
        self.z = self.sigmoid.forward(self.a)
        self.z = np.append(1, self.z)
        self.b = self.linear.forward(self.z, beta)
        self.y_hat = self.softmax.forward(self.b)
        self.J = self.cross_entropy.forward(label, self.y_hat)

    def backward(self, data, label, alpha, beta):
        self.Gy = self.cross_entropy.backward(label, self.y_hat, self.J, 1)
        self.Gb = self.softmax.backward(self.b, self.y_hat, self.Gy)
        self.G_beta, self.Gz = self.linear.backward(self.z, beta, self.Gb)
        self.Ga = self.sigmoid.backward(self.a, self.z, self.Gz)
        self.G_alpha, self.Gx = self.linear.backward(data, alpha, self.Ga)
        return self.G_alpha, self.G_beta


def random_initialize(hidden_units):
    alpha_w, alpha_h = hidden_units, 129
    beta_w, beta_h = 10, (hidden_units + 1)
    alpha = np.random.uniform(-0.1, 0.1, (alpha_w, alpha_h))
    beta = np.random.uniform(-0.1, 0.1, (beta_w, beta_h))
    return alpha, beta

def zero_initialize(hidden_units):
    alpha_w, alpha_h = hidden_units, 129
    beta_w, beta_h = 10, (hidden_units + 1)
    alpha = np.zeros((alpha_w, alpha_h))
    beta = np.zeros((beta_w, beta_h))
    return alpha, beta


def label2onehot(label):
    res = np.zeros((len(label), 10))
    res[np.arange(len(label)), label] = 1
    return res


def evaluate(data, label, alpha, beta, net):
    s = 0
    for i in range(len(data)):
        net.forward(data[i], label[i], alpha, beta)
        s += net.J
    return s / len(data)


def SGD(train_data, test_data, train_label, test_label,
        init_flag, hidden_units, num_epoch, learning_rate):
    if init_flag == '1':
        alpha, beta = random_initialize(hidden_units)
    else:
        alpha, beta = zero_initialize(hidden_units)
    train_len = len(train_data)
    train_label = label2onehot(train_label)
    test_label = label2onehot(test_label)
    net = Net()
    for e in range(num_epoch):
        for i in range(train_len):
            x, y = train_data[i], train_label[i]
            net.forward(x, y, alpha, beta)
            G_alpha, G_beta = net.backward(x, y, alpha, beta)
            alpha -= G_alpha * learning_rate
            beta -= G_beta * learning_rate
            # print(net.J)
        print(alpha, beta)
        train_ce = evaluate(train_data, train_label, alpha, beta, net)
        test_ce = evaluate(test_data, test_label, alpha, beta, net)
        print(train_ce, test_ce)
    return alpha, beta


def read_data(fn):
    f = open(fn, 'r').readlines()
    data = np.array(list(map(lambda s:s.split(','), f)))
    data = data.astype(np.int)
    labels = data[:, 0].copy()
    data[:, 0] = 1
    return labels, data


if __name__ == '__main__':
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_epoch = sys.argv[6]
    hidden_units = sys.argv[7]
    init_flag = sys.argv[8]
    learning_rate = sys.argv[9]

    train_labels, train_data = read_data(train_input)
    test_labels, test_data = read_data(test_input)
    alpha, beta = SGD(train_data, test_data, train_labels, test_labels, init_flag,
                      int(hidden_units), int(num_epoch), float(learning_rate))


