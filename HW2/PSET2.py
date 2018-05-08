import numpy as np

def one_hot(x, k):
    feature = np.zeros((k,1))
    feature[x-1] = 1
    return feature


# Takes a list of numbers and returns a column vector:  n x 1
def cv(value_list):
    return np.transpose(rv(value_list))

# Takes a list of numbers and returns a row vector: 1 x n
def rv(value_list):
    return np.array([value_list])

# x is dimension d by 1
# th is dimension d by 1
# th0 is a scalar
# return a 1 by 1 matrix
def y(x, th, th0):
   return np.dot(np.transpose(th), x) + th0

# x is dimension d by 1
# th is dimension d by 1
# th0 is dimension 1 by 1
# return 1 by 1 matrix of +1, 0, -1
def positive(x, th, th0):
   return np.sign(y(x, th, th0))

# data is dimension d by n
# labels is dimension 1 by n
# ths is dimension d by 1
# th0s is dimension 1 by 1
# return 1 by 1 matrix of integer indicating number of data points correct for
# each separator.
def score(data, labels, th, th0):
   return np.sum(positive(data, th, th0) == labels)


def eval_classifier(learner, data_train, labels_train, data_test, labels_test):
    theta, theta_0 = learner(data_train, labels_train)
    error_ = score(data_test, labels_test, theta, theta_0)
    _, n = labels_test.shape
    return error_/n


# test cases:
# eval_learning_alg(perceptron, gen_big_data(), 10, 10, 5)
# eval_learning_alg(averaged_perceptron, gen_big_data(), 10, 10, 5)
def eval_learning_alg(learner, data_gen, n_train, n_test, it):
    accuracy = 0
    for _ in range(it):
        train_data, train_labels = data_gen(n_train)
        test_data, test_labels = data_gen(n_test)
        theta, theta_0 = learner(train_data,train_labels)
        accuracy += score(test_data, test_labels, theta, theta_0)
    return accuracy/(n_test*it)


def stupid_eval_learning_alg(learner, data_gen, n_train, it):
    accuracy = 0
    for _ in range(it):
        train_data, train_labels = data_gen(n_train)
        theta, theta_0 = learner(train_data,train_labels)
        accuracy += score(train_data, train_labels, theta, theta_0)
    return accuracy/(n_train*it)
# test cases:
# xval_learning_alg(perceptron, big_data, big_data_labels, 5)
# xval_learning_alg(averaged_perceptron, big_data, big_data_labels, 5)

def xval_learning_alg(learner, data, labels, k):
    print(data.shape)
    splitted_data = np.array_split(data, k, axis=1)
    print(len(splitted_data))
    print(splitted_data[0].shape)
    splitted_labels = np.array_split(labels, k, axis=1)
    accuracy = 0
    for i in range(k):
        valid_data = splitted_data[i]
        valid_labels = splitted_labels[i]
        if i == 0:
            train_data = np.concatenate(splitted_data[i + 1:], axis=1)
            train_labels = np.concatenate(splitted_labels[i + 1:], axis=1)
        elif i == k - 1:
            train_data = np.concatenate(splitted_data[:i], axis=1)
            train_labels = np.concatenate(splitted_labels[:i], axis=1)
        else:
            train_data = np.concatenate(
                (np.concatenate(splitted_data[:i], axis=1), np.concatenate(splitted_data[i + 1:], axis=1)), axis=1)
            train_labels = np.concatenate(
                (np.concatenate(splitted_labels[:i], axis=1), np.concatenate(splitted_labels[i + 1:], axis=1)), axis=1)
        _, n = valid_labels.shape
        theta, theta_0 = learner(train_data, train_labels)
        accuracy += score(valid_data, valid_labels, theta, theta_0) / n
    return accuracy / (k)


def perceptron(data, labels, params={}, hook=None, origin=False):
    # if T not in params, default to 100
    T = params.get('T', 1000)
    d, n = data.shape
    # print('d: %d, n:%d'%(d,n))
    theta = np.zeros((d,1))
    theta_0 = np.zeros((1,1))
    flag = False
    mistake = 0
    for i in range(T):
        for j in range(n):
            x = np.reshape(data[:, j],(d,1))
            # print('x: ', x, ' with shape:', x.shape)
            # print('theta: ', theta, ' with shape:', theta.shape)
            # print('labels[iter_data]: ', labels[:,j], ' with shape: ', labels.shape)
            if (np.dot(np.transpose(theta), x) + theta_0)*labels[:, j] <= 0:
                # print('x:',x, ' and label:', labels[:, j])
                theta += labels[:, j] * x
                mistake += 1
                if not origin:
                    theta_0 = theta_0 + labels[:, j]
            error = score(data, labels, theta, theta_0)
            if error == 0:
                flag = True
                break

        if flag:
            break
    print(error)
    print(mistake)
    return (theta, theta_0)


def averaged_perceptron(data, labels, params={}, hook=None):
    # if T not in params, default to 100
    # T = params.get('T', 100)
    T = 1000
    d, n = data.shape
    theta = np.zeros((d, 1))
    theta_0 = np.zeros((1, 1))
    thetas = np.zeros((d,1))
    theta_0s = np.zeros((1,1))
    # print(d,n)
    # print(T)
    # print(n*T)
    for i in range(T):
        for j in range(n):
            x = np.reshape(data[:, j], (d, 1))
            if (np.dot(np.transpose(theta), x) + theta_0 )*labels[:, j] <= 0:
                theta += labels[:, j] * x
                theta_0 = theta_0 + labels[:, j]
            thetas += theta
            theta_0s += theta_0
    # print(theta, theta_0)
    print(i)
    return (thetas/(n*T), theta_0s/(n*T))


def calc_margin(data, labels, theta, theta_0):
    margins = (np.dot(np.transpose(theta), data)+theta_0)*labels/(np.dot(np.transpose(theta),theta))**.5
    return margins

def calc_distance(data, theta, theta_0):
    distance = (np.dot(np.transpose(theta), data)+theta_0)/(np.dot(np.transpose(theta),theta))**.5
    return distance

def convert_one_hot(data, k):
    d, n = data.shape
    new_data = np.zeros((k, n))
    for i in range(n):
        new_data[:, i] = one_hot(data[:, i], k)[:,0]
    return new_data


if __name__ == "__main__":
    data_set = np.array(([[1, 2, 3, 5]]))
    label = np.array([[1, -1, 1, 1]])
    new_data_set = convert_one_hot(data_set, 5)
    print(perceptron(new_data_set, label, origin=False))
    # theta, theta_0 = perceptron(new_data_set, label)
    # print(list(np.concatenate((theta, theta_0), axis=0).T))
    # x_1 = one_hot(1, 6)
    # x_2 = one_hot(6, 6)
    # print('Samsung: ', np.dot(np.transpose(theta),x_1)+theta_0)
    # print('Nokia: ', np.dot(np.transpose(theta),x_2)+theta_0)
    # # print(np.concatenate((x_1, x_2), axis=1).shape)
    # print(calc_distance(np.concatenate((x_1,x_2),axis=1), theta, theta_0))
