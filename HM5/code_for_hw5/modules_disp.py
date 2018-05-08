import pdb
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt

def classify(X, Y, nn, it=10000, lr=0.005):
    D = X.shape[0]
    N = X.shape[1]
    O = Y.shape[0]
    # Modifies the weights and biases
    nn.sgd(X, Y, it, lr)
    # Draw it...
    def predict(x):
        return nn.modules[-1].class_fun(nn.forward(x))[0]
    xmin, ymin = np.min(X, axis=1)-1
    xmax, ymax = np.max(X, axis=1)+1
    print(xmin,ymin,xmax,ymax)
    nax = plot_objective_2d(lambda x: predict(x), xmin, xmax, ymin, ymax)
    plot_data(X, Y, nax)
    plt.show()

    return nn

####################
# SUPPORT AND DISPLAY CODE
####################

# Takes a list of numbers and returns a row vector: 1 x n
def rv(value_list):
    return np.array([value_list])
def cv(value_list):
    return np.transpose(rv(value_list))

def tidy_plot(xmin, xmax, ymin, ymax, center = False, title = None,
                 xlabel = None, ylabel = None):
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

def plot_points(x, y, ax = None, clear = False, 
                  xmin = None, xmax = None, ymin = None, ymax = None,
                  style = 'or-', equal = False):
    padup = lambda v: v + 0.05 * abs(v)
    paddown = lambda v: v - 0.05 * abs(v)
    if ax is None:
        if xmin == None: xmin = paddown(np.min(x))
        if xmax == None: xmax = padup(np.max(x))
        if ymin == None: ymin = paddown(np.min(y))
        if ymax == None: ymax = padup(np.max(y))
        ax = tidy_plot(xmin, xmax, ymin, ymax)
        x_range = xmax - xmin; y_range = ymax - ymin
        if equal and .1 < x_range / y_range < 10:
            #ax.set_aspect('equal')
            plt.axis('equal')
            if x_range > y_range:
                ax.set_xlim((xmin, xmax))
            else:
                ax.set_ylim((ymin, ymax))
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    elif clear:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.clear()
    else:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.plot(x, y, style, markeredgewidth=0.0, linewidth = 5.0)
    # Seems to occasionally mess up the limits
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.grid(True, which='both')
    return ax

def add_ones(X):
    return np.vstack([X, np.ones(X.shape[1])])

def plot_data(data, labels, ax = None, 
                  xmin = None, xmax = None, ymin = None, ymax = None):
    # Handle 1D data
    if data.shape[0] == 1:
        data = add_ones(data)
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
    else:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    for yi in set([int(_y) for _y in set(labels.flatten().tolist())]):
        color = ['r', 'g', 'b'][yi]
        marker = ['X', 'o', 'v'][yi]
        cl = np.where(labels[0,:]==yi)
        ax.scatter(data[0,cl], data[1,cl], c = color, marker = marker, s=50,
                   edgecolors = 'none')
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.grid(True, which='both')
    return ax

def plot_objective_2d(J, xmin = -5, xmax = 5,
                      ymin = -5, ymax = 5, 
                      cmin = None, cmax = None,
                      res = 50, ax = None):
    if ax is None:
        ax = tidy_plot(xmin, xmax, ymin, ymax)
    else:
        if xmin == None:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
        else:
            ax.set_xlim((xmin, xmax))
            ax.set_ylim((ymin, ymax))

    ima = np.array([[J(cv([x1i, x2i])) \
                         for x1i in np.linspace(xmin, xmax, res)] \
                         for x2i in np.linspace(ymin, ymax, res)])
    im = ax.imshow(np.flipud(ima), interpolation = 'none',
                       extent = [xmin, xmax, ymin, ymax],
                       cmap = 'viridis')
    if cmin is not None or cmax is not None:
        if cmin is None: cmin = min(ima)
        if cmax is None: cmax = max(ima)
        im.set_clim(cmin, cmax)
    plt.colorbar(im)
    return ax

