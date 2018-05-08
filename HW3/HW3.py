import numpy as np

def length(x):
    return np.dot(np.transpose(x), x)**.5

def gamma(theta, theta_0, data, label):
    result = ((np.dot(np.transpose(theta), data) + theta_0)/length(theta))*label
    return result

def s(theta, theta_0, data, label):
    d, n = data.shape
    s_list = []
    for i in range(n):
        s_list.append(gamma(theta, theta_0, data[:, i], label[:, i]))
    return s_list

def s_sum(theta, theta_0, data, label):

    return np.sum(s(theta, theta_0, data, label))

def s_max(theta, theta_0, data, label):
    return np.max(s(theta, theta_0, data, label))

def s_min(theta, theta_0, data, label):
    return np.min(s(theta, theta_0, data, label))


def hinge_loss(theta, theta_0, data, label, gamma_ref):
    gamma_data = gamma(theta, theta_0, data, label)
    if gamma_data < gamma_ref:
        return 1 - gamma_data/gamma_ref
    return 0

if __name__ == "__main__":

    ### 1
    data = np.array([[1, 2, 1, 2, 10, 10.3, 10.5, 10.7],[1, 1, 2, 2, 2, 2, 2, 2]])
    labels = np.array([[-1, -1, 1, 1, 1, 1, 1, 1]])
    blue_th = np.array([[0, 1]]).T
    blue_th0 = -1.5
    red_th = np.array([[1, 0]]).T
    red_th0 = -2.5
    # print(length(blue_th))
    # print(gamma(blue_th, blue_th0, data[:, 0], labels[:, 0]))
    # print(s(blue_th, blue_th0, data, labels))
    # print(s_sum(blue_th, blue_th0, data, labels))
    # print([s_sum(blue_th, blue_th0, data, labels), s_min(blue_th, blue_th0, data, labels),
    #        s_max(blue_th, blue_th0, data, labels)])
    # print([s_sum(red_th, red_th0, data, labels), s_min(red_th, red_th0, data, labels),
    # s_max(red_th, red_th0, data, labels)])
    ### 2
    data = np.array([[1.1, 1, 4], [3.1, 1, 2]])
    d, n = data.shape
    labels = np.array([[1, -1, -1]])
    th = np.array([[1, 1]]).T
    th0 = -4
    gamma_ref = (2**.5/2)
    list_ = [hinge_loss(th, th0, data[:, i], labels[:, i], gamma_ref) for i in range(n)]
    print(list_)