import numpy as np
import sys


class Net:

    def forward(self, data, label, alpha, beta):
        '''
        :param data: (M + 1) * 1
        :param label: K * 1
        :param alpha: D * (M + 1)
        :param beta: K * (D + 1)
        :return:
        '''
        self.a = np.dot(alpha, data)   # D * 1
        self.z = 1 / (1 + np.exp(-self.a))   # D * 1
        self.z = np.append(1, self.z).reshape((-1, 1))    # (D + 1) * 1
        # self.z = np.vstack((np.array([[1 for _ in range(self.z.shape[1])]]), self.z))
        self.b = np.dot(beta, self.z)    # K * 1
        self.y_hat = np.exp(self.b) / np.sum(np.exp(self.b), axis=0)    # K * 1
        self.J = - np.dot(label.transpose(), np.log(self.y_hat))   # 1 * 1
        return self.J
        # print(self.a.shape, self.z.shape, self.b.shape, self.y_hat.shape, self.J.shape)

    def backward(self, data, label, alpha, beta):
        '''
        :param data: (M + 1) * 1
        :param label: K * 1
        :param alpha: D * (M + 1)
        :param beta: K * (D + 1)
        :return:
        '''
        self.Gy = - label / self.y_hat    # K * 1
        h = np.diag(self.y_hat.flatten()) - np.dot(self.y_hat, self.y_hat.transpose())   # K * K
        self.Gb = np.dot(self.Gy.transpose(), h).transpose()   # K * 1
        self.G_beta = np.dot(self.Gb, self.z.transpose())   # K * (D + 1)
        self.Gz = np.dot(self.G_beta.transpose(), self.Gb)    # (D + 1) * 1
        self.Ga = self.Gz * self.z * (1 - self.z)   # (D + 1) * 1
        self.G_alpha = np.dot(self.Ga[1:], data.transpose())   # D * (M + 1)
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
        ce = net.forward(data[i], label[i], alpha, beta)
        # print(ce)
        s += ce
    return s / len(data)


def predict(data, label, alpha, beta, net):
    num = 0
    preds = []
    for i in range(len(data)):
        net.forward(data[i], label[i], alpha, beta)
        y_hat = net.y_hat
        pred = np.argmax(y_hat)
        if pred == label[i]:
            num += 1
        preds.append(pred)
    return 1 - num / len(data), preds


def finite_diff(x, y, alpha, beta, net):
    epsilon = 1e-5
    theta = alpha
    grad = np.zeros(theta.shape)
    for m in range(len(grad)):
        for n in range(len(grad[0])):
            d = np.zeros(theta.shape)
            d[m][n] = 1
            v = net.forward(x, y, theta + epsilon * d, beta)
            v -= net.forward(x, y, theta - epsilon * d, beta)
            v /= 2*epsilon
            grad[m][n] = v
    # print('grad = ', grad)
    return grad


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
            x = x.reshape((-1, 1))
            y = y.reshape((-1, 1))
            net.forward(x, y, alpha, beta)
            G_alpha, G_beta = net.backward(x, y, alpha, beta)
            alpha -= G_alpha * learning_rate
            beta -= G_beta * learning_rate
            # print('G_alpha = ', G_alpha)
            # print('G_beta = ', G_beta)
            # G_alpha = finite_diff(x, y, alpha, beta, net)
            # alpha -= G_alpha * learning_rate
            # print(net.J)
        # print(alpha, beta)
        train_ce = evaluate(train_data, train_label, alpha, beta, net)
        test_ce = evaluate(test_data, test_label, alpha, beta, net)
        print(train_ce, test_ce)
    return alpha, beta, net


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
    alpha, beta, net = SGD(train_data, test_data, train_labels, test_labels, init_flag,
                      int(hidden_units), int(num_epoch), float(learning_rate))
    train_error_rate, train_pred = predict(train_data, train_labels, alpha, beta, net)
    test_error_rate, test_pred = predict(test_data, test_labels, alpha, beta, net)
    print(train_error_rate, test_error_rate)


