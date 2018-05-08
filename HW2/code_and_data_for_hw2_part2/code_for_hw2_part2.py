# Implement perceptron, average perceptron, and pegasos
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pdb
Global_T = 1
######################################################################
# Plotting

def tidy_plot(xmin, xmax, ymin, ymax, center = False, title = None,
                 xlabel = None, ylabel = None):
    plt.ion()
    plt.figure(facecolor="white")
    ax = plt.subplot()
    if center:
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_smart_bounds(True)
        ax.spines['bottom'].set_smart_bounds(True)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    else:
        ax.spines["top"].set_visible(False)    
        ax.spines["right"].set_visible(False)    
        ax.get_xaxis().tick_bottom()  
        ax.get_yaxis().tick_left()
    eps = .05
    plt.xlim(xmin-eps, xmax+eps)
    plt.ylim(ymin-eps, ymax+eps)
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    return ax

def plot_separator(ax, th, th_0):
    xmin, xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    pts = []
    eps = 1.0e-6
    # xmin boundary crossing is when xmin th[0] + y th[1] + th_0 = 0
    # that is, y = (-th_0 - xmin th[0]) / th[1]
    if abs(th[1,0]) > eps:
        pts += [np.array([x, (-th_0 - x * th[0,0]) / th[1,0]]) \
                                                        for x in (xmin, xmax)]
    if abs(th[0,0]) > 1.0e-6:
        pts += [np.array([(-th_0 - y * th[1,0]) / th[0,0], y]) \
                                                         for y in (ymin, ymax)]
    in_pts = []
    for p in pts:
        if (xmin-eps) <= p[0] <= (xmax+eps) and \
           (ymin-eps) <= p[1] <= (ymax+eps):
            duplicate = False
            for p1 in in_pts:
                if np.max(np.abs(p - p1)) < 1.0e-6:
                    duplicate = True
            if not duplicate:
                in_pts.append(p)
    if in_pts and len(in_pts) >= 2:
        # Plot separator
        vpts = np.vstack(in_pts)
        ax.plot(vpts[:,0], vpts[:,1], 'k-', lw=2)
        # Plot normal
        vmid = 0.5*(in_pts[0] + in_pts[1])
        scale = np.sum(th*th)**0.5
        diff = in_pts[0] - in_pts[1]
        dist = max(xmax-xmin, ymax-ymin)
        vnrm = vmid + (dist/10)*(th.T[0]/scale)
        vpts = np.vstack([vmid, vnrm])
        ax.plot(vpts[:,0], vpts[:,1], 'k-', lw=2)
        # Try to keep limits from moving around
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
    else:
        print('Separator not in plot range')

def plot_data(data, labels, ax = None, clear = False,
                  xmin = None, xmax = None, ymin = None, ymax = None):
    if ax is None:
        if xmin == None: xmin = np.min(data[0, :]) - 0.5
        if xmax == None: xmax = np.max(data[0, :]) + 0.5
        if ymin == None: ymin = np.min(data[1, :]) - 0.5
        if ymax == None: ymax = np.max(data[1, :]) + 0.5
        ax = tidy_plot(xmin, xmax, ymin, ymax)

        x_range = xmax - xmin; y_range = ymax - ymin
        if .1 < x_range / y_range < 10:
            ax.set_aspect('equal')
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    elif clear:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.clear()
    else:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    colors = np.choose(labels > 0, cv(['r', 'g']))[0]
    ax.scatter(data[0,:], data[1,:], c = colors,
                    marker = 'o', s=50, edgecolors = 'none')
    # Seems to occasionally mess up the limits
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.grid(True, which='both')
    #ax.axhline(y=0, color='k')
    #ax.axvline(x=0, color='k')
    return ax

######################################################################
#   Utilities

# Takes a list of numbers and returns a column vector:  n x 1
def cv(value_list):
    return np.transpose(rv(value_list))

# Takes a list of numbers and returns a row vector: 1 x n
def rv(value_list):
    return np.array([value_list])

# x is dimension d by n
# th is dimension d by m
# th0 is dimension 1 by m
# return matrix of y values for each column of x and theta: dimension m x n
def y(x, th, th0):
    return np.dot(np.transpose(th), x) + np.transpose(th0)

def length(d_by_m):
    return np.sum(d_by_m * d_by_m, axis = 0, keepdims = True)**0.5

# x is dimension d by n
# th is dimension d by m
# th0 is dimension 1 by m
# return matrix of signed dist for each column of x and theta: dimension m x n
def signed_dist(x, th, th0):
    # ugly transpose, unmotivated
    return y(x, th, th0) / np.transpose(length(th))

# x is dimension d by n
# th is dimension d by m
# th0 is dimension 1 by m
# return matrix of +1, 0, -1 for each column of x and theta:
#                                                dimension m x n
def positive(x, theta, theta_0):
    return np.sign(y(x, theta, theta_0))

def score(data, labels, th, th_0):
    return np.sum(positive(data, th, th_0) == labels, axis = 1, keepdims = True)

def xval_learning_alg(learner, data, labels, k, params={}):
    s_data = np.array_split(data, k, axis=1)
    s_labels = np.array_split(labels, k, axis=1)

    score_sum = 0
    for i in range(k):
        data_train = np.concatenate(s_data[:i] + s_data[i+1:], axis=1)
        labels_train = np.concatenate(s_labels[:i] + s_labels[i+1:], axis=1)
        data_test = np.array(s_data[i])
        labels_test = np.array(s_labels[i])
        score_sum += eval_classifier(learner, data_train, labels_train,
                                              data_test, labels_test, params)
    return score_sum/k

def eval_classifier(learner, data_train, labels_train, data_test, labels_test, params={}):
    th, th0 = learner(data_train, labels_train, params)
    return score(data_test, labels_test, th, th0)/data_test.shape[1]


def averaged_perceptron(data, labels, params={}, hook=None):
    # if T not in params, default to 100
    T = params.get('T', Global_T)
    d, n = data.shape
    theta = np.zeros((d, 1))
    theta_0 = np.zeros((1, 1))
    thetas = np.zeros((d,1))
    theta_0s = np.zeros((1,1))
    # print(d,n)
    # print(T)
    # print(n*T)
    m = 0
    for i in range(T):
        for j in range(n):
            x = np.reshape(data[:, j], (d, 1))
            if (np.dot(np.transpose(theta), x) + theta_0 )*labels[:, j] <= 0:
                m += 1
                theta += labels[:, j] * x
                theta_0 = theta_0 + labels[:, j]
            thetas += theta
            theta_0s += theta_0
    # print(theta, theta_0)
    # print("number of mistakes: %d" % m)
    return (thetas/(n*T), theta_0s/(n*T))


def perceptron(data, labels, params = {}, hook = None):
    T = params.get('T', Global_T)
    (d, n) = data.shape
    m = 0
    theta = np.zeros((d, 1)); theta_0 = np.zeros((1, 1))
    for t in range(T):
        for i in range(n):
            x = data[:,i:i+1]
            y = labels[:,i:i+1]
            if y * positive(x, theta, theta_0) <= 0.0:
                m += 1
                theta = theta + y * x
                theta_0 = theta_0 + y
                if hook: hook((theta, theta_0))
    # print("number of mistakes: %d"%m)
    return theta, theta_0

######################################################################
#   Tests

def test_linear_classifier(dataFun, learner, learner_params = {},
                             draw = True, refresh = True, pause = True):
    data, labels = dataFun()
    d, n = data.shape
    if draw:
        ax = plot_data(data, labels)
        def hook(params):
            (th, th0) = params
            if refresh: plot_data(data, labels, ax, clear = True)
            plot_separator(ax, th, th0)
            print('th', th.T, 'th0', th0)
            if pause: input('go?')
    else:
        hook = None
    th, th0 = learner(data, labels, hook = hook, params = learner_params)
    print("Final score", float(score(data, labels, th, th0)) / n)
    print("Params", np.transpose(th), th0)
    
######################################################################
# For auto dataset

def load_auto_data(path_data):
    """
    Returns a list of dict with keys:
    """
    numeric_fields = {'mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                      'acceleration', 'model_year', 'origin'}
    data = []
    with open(path_data) as f_data:
        for datum in csv.DictReader(f_data, delimiter='\t'):
            for field in list(datum.keys()):
                if field in numeric_fields and datum[field]:
                    datum[field] = float(datum[field])
            data.append(datum)
    return data

# Feature transformations
def std_vals(data, f):
    vals = [entry[f] for entry in data]
    avg = sum(vals)/len(vals)
    dev = [(entry[f] - avg)**2 for entry in data]
    sd = (sum(dev)/len(vals))**0.5
    return (avg, sd)

def standard(v, std):
    return [(v-std[0])/std[1]]

def raw(x):
    return [x]

def one_hot(v, entries):
    vec = len(entries)*[0]
    vec[entries.index(v)] = 1
    return vec

# The class (mpg) added to the front of features
def auto_data_and_labels(auto_data, features):
    features = [('mpg', raw)] + features
    std = {f:std_vals(auto_data, f) for (f, phi) in features if phi==standard}
    entries = {f:list(set([entry[f] for entry in auto_data])) \
               for (f, phi) in features if phi==one_hot}
    print('avg and std', std)
    print('entries in one_hot field', entries)
    vals = []
    for entry in auto_data:
        phis = []
        for (f, phi) in features:
            if phi == standard:
                phis.extend(phi(entry[f], std[f]))
            elif phi == one_hot:
                phis.extend(phi(entry[f], entries[f]))
            else:
                phis.extend(phi(entry[f]))
        vals.append(np.array([phis]))
    data_labels = np.vstack(vals)
    np.random.seed(0)
    np.random.shuffle(data_labels)
    return data_labels[:, 1:].T, data_labels[:, 0:1].T

######################################################################
# For food review dataset

from string import punctuation, digits, printable
import csv

def load_review_data(path_data):
    """
    Returns a list of dict with keys:
    * sentiment: +1 or -1 if the review was positive or negative, respectively
    * text: the text of the review
    """
    basic_fields = {'sentiment', 'text'}
    data = []
    with open(path_data) as f_data:
        for datum in csv.DictReader(f_data, delimiter='\t'):
            for field in list(datum.keys()):
                if field not in basic_fields:
                    del datum[field]
            if datum['sentiment']:
                datum['sentiment'] = int(datum['sentiment'])
            data.append(datum)
    return data

printable = set(printable)
def clean(s):
    return filter(lambda x: x in printable, s)

def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    # return [ps.stem(w) for w in input_string.lower().split()]
    return input_string.lower().split()

def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Section 3 (e.g. remove stopwords, add bigrams etc.)
    """
    dictionary = {} # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in dictionary:
                dictionary[word] = len(dictionary)
    return dictionary

def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.
    """

    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] = 1
    # We want the feature vectors as columns
    return feature_matrix.T

print('code for part2 loaded')

if __name__ == "__main__":
    data = load_auto_data('auto-mpg.tsv')
    feature_set_1 = [('cylinders',raw), ('displacement',raw), ('horsepower', raw),
                                                  ('weight',raw), ('acceleration',raw), ('origin',raw)]
    feature_set_2 = [('cylinders',one_hot), ('displacement',standard), ('horsepower', standard),
                                                  ('weight',standard), ('acceleration',standard), ('origin',one_hot)]
    feature_set_3 = [('weight', standard), ('origin',one_hot)]
    # feature, labels = auto_data_and_labels(data, [('cylinders',raw), ('displacement',raw), ('horsepower', raw),
    #                                               ('weight',raw), ('acceleration',raw), ('origin',raw)])

    # accuracy = xval_learning_alg(perceptron, feature, labels, k = 10)
    # print(accuracy)
    # accuracy_avg = xval_learning_alg(averaged_perceptron, feature, labels, k = 10)
    # print(accuracy_avg)
    # feature_count = 1
    # for feature in [feature_set_1, feature_set_2, feature_set_3]:
    #     print('Accuracy on features: feature_set_%d'%feature_count)
    #     for T in [50]:#, 10, 50]:
    #         print('T: %d'%T)
    #         feature, labels = auto_data_and_labels(data, feature)
    #         print('In perceptron:')
    #         accuracy = xval_learning_alg(perceptron, feature, labels, k=10, params={'T': T})
    #         print('In avg perceptron:')
    #         accuracy_avg = xval_learning_alg(averaged_perceptron, feature, labels, k=10, params={'T': T})
    #         print(list(np.concatenate([accuracy, accuracy_avg]).reshape(1,2)))
    #
    #     feature_count += 1
    # feature, labels = auto_data_and_labels(data, feature_set_2)
    # theta, theta_0 = averaged_perceptron(feature, labels, params={'T':10})
    # print(theta)
    # labels = data['mpg']
    # print(labels.shape)
    review_data = load_review_data('reviews.tsv')
    # print(data[:2])
    review_texts, review_label_list = zip(*((sample['text'], sample['sentiment']) for sample in review_data))
    dictionary = bag_of_words(review_texts)
    review_bow_feature = extract_bow_feature_vectors(review_texts, dictionary)
    review_labels = rv(review_label_list)
    # for T in [1, 10, 20]:
    #     accuracy = xval_learning_alg(perceptron, review_bow_feature, review_labels, k=10, params={'T': T})
    #     accuracy_avg = xval_learning_alg(averaged_perceptron, review_bow_feature, review_labels, k=10, params={'T': T})
    #     print(list(np.concatenate([accuracy, accuracy_avg]).reshape(1,2)))
    #
    #
    theta, theta_0 = averaged_perceptron(review_bow_feature, review_labels, {'T': 10})
    biggest_coef = np.sort(theta.reshape(theta.shape[0]))[:10]
    indxs = []
    for coef in biggest_coef:
        indxs.append(np.where(theta == coef)[0][0])
    print(indxs)
    words = [list(dictionary.keys())[list(dictionary.values()).index(val)] for val in indxs]
    print(words)
    # max = []
    # for index, coef in enumerate(theta):

