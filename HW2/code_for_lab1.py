# Implement perceptron, average perceptron, and pegasos
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pdb
import itertools
import operator


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

# LPK: replace this with something that will work even for vertical lines
#  and goes all the way to the boundaries
# Also draw a little normal vector in the positive direction
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

# Must either specify limits or existing ax
def plot_nonlin_sep(predictor, ax = None, xmin = None , xmax = None,
                        ymin = None, ymax = None, res = 30):
    if ax is None:
        ax = tidy_plot(xmin, xmax, ymin, ymax)
    else:
        if xmin == None:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
        else:
            ax.set_xlim((xmin, xmax))
            ax.set_ylim((ymin, ymax))

    cmap = colors.ListedColormap(['black', 'white'])
    bounds=[-2,0,2]
    norm = colors.BoundaryNorm(bounds, cmap.N)            
            
    ima = np.array([[predictor(x1i, x2i) \
                         for x1i in np.linspace(xmin, xmax, res)] \
                         for x2i in np.linspace(ymin, ymax, res)])
    im = ax.imshow(np.flipud(ima), interpolation = 'none',
                       extent = [xmin, xmax, ymin, ymax],
                       cmap = cmap, norm = norm)

######################################################################
#   Utilities

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

######################################################################
#   Data Sets

# Return d = 2 by n = 4 data matrix and 1 x n = 4 label matrix
def super_simple_separable_through_origin():
    X = np.array([[2, 3, 9, 12],
                  [5, 1, 6, 5]])
    y = np.array([[1, -1, 1, -1]])
    return X, y

def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, -1, 1, -1]])
    return X, y

def xor():
    X = np.array([[1, 2, 1, 2],
                  [1, 2, 2, 1]])
    y = np.array([[1, 1, -1, -1]])
    return X, y

def xor_more():
    X = np.array([[1, 2, 1, 2, 2, 4, 1, 3],
                  [1, 2, 2, 1, 3, 1, 3, 3]])
    y = np.array([[1, 1, -1, -1, 1, 1, -1, -1]])
    return X, y

# Test data for problem 2.1
data1, labels1, data2, labels2 = \
(np.array([[-2.97797707,  2.84547604,  3.60537239, -1.72914799, -2.51139524,
         3.10363716,  2.13434789,  1.61328413,  2.10491257, -3.87099125,
         3.69972003, -0.23572183, -4.19729119, -3.51229538, -1.75975746,
        -4.93242615,  2.16880073, -4.34923279, -0.76154262,  3.04879591,
        -4.70503877,  0.25768309,  2.87336016,  3.11875861, -1.58542576,
        -1.00326657,  3.62331703, -4.97864369, -3.31037331, -1.16371314],
       [ 0.99951218, -3.69531043, -4.65329654,  2.01907382,  0.31689211,
         2.4843758 , -3.47935105, -4.31857472, -0.11863976,  0.34441625,
         0.77851176,  1.6403079 , -0.57558913, -3.62293005, -2.9638734 ,
        -2.80071438,  2.82523704,  2.07860509,  0.23992709,  4.790368  ,
        -2.33037832,  2.28365246, -1.27955206, -0.16325247,  2.75740801,
         4.48727808,  1.6663558 ,  2.34395397,  1.45874837, -4.80999977]]), np.array([[-1., -1., -1., -1., -1., -1.,  1.,  1.,  1., -1., -1., -1., -1.,
        -1.,  1., -1.,  1., -1., -1., -1.,  1.,  1.,  1.,  1.,  1., -1.,
        -1., -1., -1., -1.]]), np.array([[ 0.6894022 , -4.34035772,  3.8811067 ,  4.29658177,  1.79692041,
         0.44275816, -3.12150658,  1.18263462, -1.25872232,  4.33582168,
         1.48141202,  1.71791177,  4.31573568,  1.69988085, -2.67875489,
        -2.44165649, -2.75008176, -4.19299345, -3.15999758,  2.24949368,
         4.98930636, -3.56829885, -2.79278501, -2.21547048,  2.4705776 ,
         4.80481986,  2.77995092,  1.95142828,  4.48454942, -4.22151738],
       [-2.89934727,  1.65478851,  2.99375325,  1.38341854, -4.66701003,
        -2.14807131, -4.14811829,  3.75270334,  4.54721208,  2.28412663,
        -4.74733482,  2.55610647,  3.91806508, -2.3478982 ,  4.31366925,
        -0.92428271, -0.84831235, -3.02079092,  4.85660032, -1.86705397,
        -3.20974025, -4.88505017,  3.01645974,  0.03879148, -0.31871427,
         2.79448951, -2.16504256, -3.91635569,  3.81750006,  4.40719702]]), np.array([[-1., -1.,  1.,  1., -1., -1., -1.,  1.,  1.,  1., -1.,  1.,  1.,
        -1.,  1.,  1.,  1., -1., -1., -1.,  1., -1.,  1., -1.,  1., -1.,
        -1.,  1.,  1.,  1.]]))

big_data, big_data_labels = (np.array([[-0.82977995,  2.20324493, -4.99885625, -1.97667427, -3.53244109,
        -4.07661405, -3.13739789, -1.54439273, -1.03232526,  0.38816734,
        -0.80805486,  1.852195  , -2.9554775 ,  3.78117436, -4.72612407,
         1.7046751 , -0.82695198,  0.58689828, -3.59613061, -3.01898511,
         3.00744569,  4.68261576, -1.86575822,  1.92322616,  3.76389152,
         3.94606664, -4.14955789, -4.60945217, -3.3016958 ,  3.78142503,
        -4.01653166, -0.78892375,  4.5788953 ,  0.33165285,  1.91877114,
        -1.84484369,  1.86500928,  3.34625672, -4.81711723,  2.50144315,
         4.88861089,  2.48165654, -2.19556008,  2.89279328, -3.96773993,
        -0.52106474,  4.08595503, -2.06385852, -2.12224661, -3.69971428,
        -4.80633042,  1.78835533, -2.88371884, -2.34453341, -0.08426841,
        -4.46637455,  0.74117605, -3.53271425,  0.89305537,  1.9975836 ,
        -3.97665571, -0.85944012,  1.94400158, -0.8582073 , -4.50046541,
         0.35896406,  1.63794645,  0.14889112,  4.44594756,  0.86555041,
         4.03401915, -3.62525296, -3.60723653,  3.07391289, -1.02323163,
        -3.34645803,  4.2750858 , -1.5223414 ,  2.50812103,  2.25997985,
         3.83306091,  1.23672207,  2.50942434, -1.51101658, -2.30072108,
         3.95886218, -0.7190881 ,  4.64840047,  1.63441498,  1.2169572 ,
        -3.85254027,  4.49489259, -0.50087867,  0.78389614, -0.91863197,
        -2.6297302 ,  4.03379521,  0.73679487, -4.97129673,  1.17144914],
       [-1.73355098,  0.27058102,  3.85942099, -1.4273024 ,  4.08535151,
         1.23360116, -4.84178757,  4.29437234,  1.90896918,  4.9732285 ,
        -3.27659492, -3.6286425 ,  4.32595463,  1.96818161, -4.33999827,
         2.55463053,  2.53876188,  4.23024536,  2.11524759, -3.75729038,
        -4.80119866, -4.73789013, -4.71693512, -2.53788932,  3.60027949,
         0.38831064,  0.52821979,  3.42030892, -3.75826685, -2.20816321,
         0.85759271,  4.69595748,  0.61030219, -4.81352711,  3.00632673,
        -2.67025726,  3.07105196, -1.12139356,  3.63541855,  2.47121643,
         0.56240234, -3.63544774, -4.4008231 , -3.78656544, -4.55448121,
        -3.92505871, -2.74290661,  2.1298898 ,  0.59716982, -4.8744402 ,
        -4.2802572 ,  4.6727633 ,  0.68100462, -2.96706765, -2.47674255,
         2.43825854, -3.04570519,  0.81358927,  4.70019989,  3.46828801,
        -2.60152241, -0.06230286,  1.19955718,  3.289809  , -3.43208605,
        -4.81423798, -4.29977856, -0.13654889,  1.06329462,  0.68851437,
        -1.82637591,  4.88616154,  0.79745219, -1.19858827,  0.50948219,
         2.45334431,  1.69232893, -2.35080442, -4.33665166, -1.29915802,
         1.29717507, -2.8982599 ,  2.52755554, -4.33463519, -2.39684901,
         3.04754564, -3.06565717,  1.39460881,  0.24670309,  4.2480797 ,
        -2.3670323 , -4.34038909,  2.35065963,  2.7217803 ,  4.07815853,
         4.31972069, -4.86048427, -2.65637914,  1.16778357,  4.49016321]]), np.array([[-1.,  1.,  1., -1.,  1., -1., -1.,  1.,  1.,  1., -1., -1., -1.,
        -1.,  1.,  1., -1.,  1., -1.,  1.,  1., -1.,  1., -1.,  1.,  1.,
        -1., -1., -1.,  1., -1.,  1.,  1., -1.,  1., -1., -1.,  1.,  1.,
         1.,  1., -1., -1.,  1., -1., -1., -1.,  1., -1., -1.,  1.,  1.,
        -1., -1., -1., -1., -1., -1.,  1.,  1., -1., -1.,  1.,  1., -1.,
        -1., -1., -1.,  1.,  1.,  1.,  1., -1.,  1., -1., -1., -1., -1.,
        -1.,  1.,  1., -1.,  1., -1., -1.,  1., -1.,  1.,  1.,  1., -1.,
        -1.,  1.,  1.,  1., -1., -1.,  1., -1., -1.]]))

# Test data for problem 2.2
big_data, big_data_labels = (np.array([[-2.04297103, -1.85361169, -2.65467827, -1.23013149, -0.31934782,
         1.33112127,  2.3297942 ,  1.47705445, -1.9733787 , -2.35476882,
        -4.97193554,  3.49851995,  4.00302943,  0.83369183,  0.41371989,
         4.37614714,  1.03536965,  1.2354608 , -0.7933465 , -3.85456759,
         3.22134658, -3.39787483, -1.31182253, -2.61363628, -1.14618119,
        -0.2174626 ,  1.32549116,  2.54520221,  0.31565661,  2.24648287,
        -3.33355258, -0.98689271, -0.24876636, -3.16008017,  1.22353111,
         4.77766994, -1.81670773, -3.58939471, -2.16268851,  2.88028351,
        -3.42297827, -2.74992813, -0.40293356, -3.45377267,  0.62400624,
        -0.35794507, -4.1648704 , -1.08734116,  0.22367444,  1.09067619,
         1.28738004,  2.07442478,  4.61951855,  4.47029706,  2.86510481,
         4.12532285,  0.48170777,  0.60089857,  4.50287515,  2.95549453,
         4.22791451, -1.28022286,  2.53126681,  2.41887277, -4.9921717 ,
         4.15022718,  0.49670572,  2.0268248 , -4.63475897, -4.20528418,
         1.77013481, -3.45389325,  1.0238472 , -1.2735185 ,  4.75384686,
         1.32622048, -0.13092625,  1.23457116, -1.69515197,  2.82027615,
        -1.01140935,  3.36451016,  4.43762708, -4.2679604 ,  4.76734154,
        -4.14496071, -4.38737405, -1.13214501, -2.89008477,  3.22986894,
         1.84103699, -3.91906092, -2.8867831 ,  2.31059245, -3.62773189,
        -4.58459406, -4.06343392, -3.10927054,  1.09152472,  2.99896855],
       [-2.1071566 , -3.06450052, -3.43898434,  0.71320285,  1.51214693,
         4.14295175,  4.73681233, -2.80366981,  1.56182223,  0.07061724,
        -0.92053415, -3.61953464,  0.39577344, -3.03202474, -4.90408303,
        -0.10239158, -1.35546287,  1.31372748, -1.97924525, -3.72545813,
         1.84834303, -0.13679709,  1.36748822, -2.92886952, -2.48367819,
        -0.0894489 , -2.99090327,  0.35494698,  0.94797491,  4.20393035,
        -3.14009852, -4.86292242,  3.2964068 , -0.9911453 ,  4.39465   ,
         3.64956975, -0.72225648, -0.15864119, -2.0340774 , -4.00758749,
         0.8627915 ,  3.73237594, -0.70011824,  1.07566463, -4.05063547,
        -3.98137177,  4.82410619,  2.5905222 ,  0.34188269, -1.44737803,
         3.27583966,  2.06616486, -4.43584161,  0.27795053,  4.37207651,
        -4.48564119,  0.7183541 ,  1.59374552, -0.13951634,  0.67825519,
        -4.02423434,  4.15893861, -1.52110278,  2.1320374 ,  3.31118893,
        -4.04072252,  2.41403912, -1.04635499,  3.39575642,  2.2189097 ,
         4.78827245,  1.19808069,  3.10299723,  0.18927394,  0.14437543,
        -4.17561642,  0.6060279 ,  0.22693751, -3.39593567,  1.14579319,
         3.65449494, -1.27240159,  0.73111639,  3.48806017,  2.48538719,
        -1.83892096,  1.42819622, -1.37538641,  3.4022984 ,  0.82757044,
        -3.81792516,  2.77707152, -1.49241173,  2.71063994, -3.33495679,
        -4.00845675,  0.719904  , -2.3257032 ,  1.65515972, -1.90859948]]), np.array([[-1., -1., -1.,  1.,  1., -1.,  1., -1.,  1., -1., -1., -1.,  1.,
         1., -1.,  1., -1., -1., -1.,  1., -1., -1.,  1., -1.,  1., -1.,
        -1.,  1.,  1., -1., -1., -1.,  1., -1.,  1.,  1.,  1., -1., -1.,
         1., -1.,  1., -1.,  1., -1.,  1.,  1.,  1.,  1., -1.,  1.,  1.,
        -1.,  1.,  1., -1., -1.,  1.,  1.,  1., -1.,  1.,  1., -1., -1.,
         1.,  1.,  1.,  1., -1.,  1., -1.,  1., -1., -1., -1.,  1.,  1.,
        -1.,  1.,  1.,  1.,  1., -1., -1.,  1., -1., -1., -1.,  1., -1.,
        -1., -1.,  1., -1., -1., -1., -1.,  1.,  1.]]))

def gen_big_data():
    nd = big_data.shape[1]
    current = [0]
    def f(n):
        for i in range(10):
            cur = current[0]
            vals = big_data[:,cur:cur+n], big_data_labels[:,cur:cur+n]
            current[0] += n
            if current[0] >= nd: current[0] = 0
            return vals
    return f

def gen_lin_separable(num_points=20, th=np.array([[3],[4]]), th_0=np.array([[0]]), dim=2):
    X = np.random.uniform(low=-5, high=5, size=(dim, num_points))
    y = np.sign(np.dot(np.transpose(th), X) + th_0)
    return X, y

def big_higher_dim_separable():
    X, y = gen_lin_separable(num_points=50, dim=6, th=np.array([[3],[4],[2],[1],[0],[3]]))
    return X, y

# Generate difficult (usually not linearly separable) data sets by
# "flipping" labels with some probability.
def gen_flipped_lin_separable(num_points=20, pflip=0.25, th=np.array([[3],[4]]), th_0=np.array([[0]]), dim=2):
    X, y = gen_lin_separable(num_points, th, th_0, dim)
    flip = np.random.uniform(low=0, high=1, size=(num_points,))
    for i in range(num_points):
        if flip[i] < pflip: y[0,i] = -y[0,i]
    return X, y

######################################################################
#   tests

def test_linear_classifier(dataFun, learner, learner_params = {},
                             draw = False, refresh = False, pause = False):
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


# test cases:
# eval_classifier(perceptron, data1, labels1, data2, labels2)
# eval_classifier(perceptron, data2, labels2, data2, labels2)

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


# test cases:
# for datafn in (super_simple_separable, xor, xor_more, big_higher_dim_separable):
#   data, labels = datafn()
#   perceptron(data, labels, {"T": 100})

def perceptron(data, labels, params={}, hook=None, origin=False):
    # if T not in params, default to 100
    T = params.get('T', 100)
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
    print(mistake)
    return (theta, theta_0)

# test cases:
# for datafn in (super_simple_separable, xor, xor_more, big_higher_dim_separable):
#   data, labels = datafn()
#   averaged_perceptron(data, labels, {"T": 100})
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
    print(theta)
    print(data)
    print((np.linalg.norm(theta)))
    margins = (np.dot(np.transpose(theta), data)+theta_0)*labels/(np.dot(np.transpose(theta),theta))**.5
    return margins

if __name__ == "__main__":
    #
    # accuracy_percp = stupid_eval_learning_alg(perceptron, gen_flipped_lin_separable, 20, 100)
    # print(accuracy_percp)
    # accuracy_avg = stupid_eval_learning_alg(averaged_perceptron, gen_flipped_lin_separable, 20, 100)
    # print(accuracy_avg)
    data_set = np.array(([[2, 3,  4,  5]]))
    # plt.plot(data_set[0], data_set[1], 'o')
    # plt.show()
    # data_set[0] = data_set[0]*.001
    # data_set = np.concatenate((data_set, np.ones((1,4))), axis=0)
    # print(data_set)
    label = np.array([[1, 1, -1, -1]])
    # theta = np.array([[0],[1],[-0.5]])
    # theta_0 = np.zeros((1,1))
    # print(calc_margin(data_set, label, theta, theta_0))
    print(perceptron(data_set, label, origin=False))
    # test_linear_classifier(xor, perceptron, learner_params = {'T':100}, draw = True, refresh = True, pause = True)
    # data_split = np.array_split(data1, 5, axis=1)
    # print(len(data_split))
    # print(data_split[0].shape)
    # part_1 = np.concatenate(data_split[:2], axis=1)
    # print(part_1.shape)
    # part_2 = np.concatenate(data_split[3:4], axis=1)
    # print(part_2.shape)
    # all = np.concatenate((part_1, part_2), axis=1)
    # print(all.shape)
    #
    # concat = np.concatenate((data_split[0:4], data_split[5:]), axis = 1)
    # print('concat shape', concat.shape)