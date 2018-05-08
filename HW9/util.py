import numpy as np

# Takes a list of numbers and returns a column vector:  n x 1
def cv(value_list):
    return rv(value_list).T

# Takes a list of numbers and returns a row vector: 1 x n
def rv(value_list):
    return np.array([value_list])

def argmax(l, f):
    """
    @param l: C{List} of items
    @param f: C{Procedure} that maps an item into a numeric score
    @returns: the element of C{l} that has the highest score
    """
    vals = [f(x) for x in l]
    return l[vals.index(max(vals))]

def argmax_with_val(l, f):
    """
    @param l: C{List} of items
    @param f: C{Procedure} that maps an item into a numeric score
    @returns: the element of C{l} that has the highest score and the score
    """
    best = l[0]; bestScore = f(best)
    for x in l:
        xScore = f(x)
        if xScore > bestScore:
            best, bestScore = x, xScore
    return (best, bestScore)