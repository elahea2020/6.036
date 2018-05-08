import numpy as np


def calc_error(data_set, label, perceptron):
    """
    perceptron is a list of [theta, theta_0]
    :param data_set:
    :param label:
    :param perceptron:
    :return:
    """
    theta, theta_0 = perceptron
    error = 0
    pred = np.sign(np.dot(np.transpose(theta), data_set) + theta_0)

    error = np.sum(pred==label,axis=1)[0]
    return error


def perceptron_learning(data_set, label, training_step=100, initializer=None, dim=2):
    """
    returns a list of [theta, theta_0]
    :param data_set:
    :param label:
    :param training_step:
    :param initializer:
    :return:
    """
    if initializer == None:
        theta = np.zeros((dim,1))
        theta_0 = 0

    iter_data = 3
    while True:
        theta += label[iter_data] * np.reshape(data_set[:,iter_data],(dim,1))
        print('theta:', theta)
        theta_0 = theta_0 + label[iter_data]
        iter_data += 1
        if iter_data >= data_set.shape[1]:
            iter_data = 0
        error = calc_error(data_set, label, [theta, theta_0])
        print('error is :', error)
        if error == 0:
            break

    return [theta, theta_0]

if __name__ == "__main__":
    data_set = np.transpose(np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]))
    label = [-1, -1, -1, -1, -1, -1, -1, 1]
    data_set_1 = np.transpose(np.array([[1,1],[2,2],[1,-1], [-1, 1]]))
    print(data_set_1.shape)
    label_1= [-1, -1, 1, 1]
    # print(data_set.shape)
    # print(perceptron_learning(data_set, label, dim=3))
    print(perceptron_learning(data_set_1, label_1))
