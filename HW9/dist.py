import pdb
import numpy as np
import random

class DDist:
    """Discrete distribution represented as a dictionary.  Can be
    sparse, in the sense that elements that are not explicitly
    contained in the dictionary are assuemd to have zero probability."""
    def __init__(self, dictionary, name = None):
        self.d = dictionary
        """ Dictionary whose keys are elements of the domain and values
        are their probabilities. """

    def prob(self, elt):
        """
        @returns: the probability associated with C{elt}
        """
        return self.d.get(elt, 0)

    def setProb(self, elt, p):
        """
        @param elt: element of the domain
        @param p: probability
        Sets probability of C{elt} to be C{p}
        """
        self.d[elt] = p

    def support(self):
        """
        @returns: A list (in any order) of the elements of this
        distribution with non-zero probabability.
        """
        return self.d.keys()

    def maxProbElt(self):
        """
        @returns: The element in this domain with maximum probability
        """
        bestP = 0
        bestElt = None
        for (elt, p) in self.d.items():
            if p > bestP:
                bestP = p
                bestElt = elt
        return (bestElt, bestP)

    def draw(self):
        """
        @returns: a randomly drawn element from the distribution
        """
        r = random.random()
        sum = 0.0
        for val in self.support():
            sum += self.prob(val)
            if r < sum:
                return val
        raise Exception('Failed to draw from '+ str(self))

    def addProb(self, val, p):
        """
        Increase the probability of element C{val} by C{p}
        """
        self.setProb(val, self.prob(val) + p)

    def mulProb(self, val, p):
        """
        Multiply the probability of element C{val} by C{p}
        """
        self.setProb(val, self.prob(val) * p)

    def expectation(self, f):
        return sum(self.prob(x) * f(x) for x in self.support())

    def normalize(self):
        """
        Divides all probabilities through by the sum of the values to
        ensure the distribution is normalized.

        Changes the distribution!!  (And returns it, for good measure)

        Generates an error if the sum of the current probability
        values is zero.
        """
        z = sum([self.prob(e) for e in self.support()])
        assert z > 0.0, 'degenerate distribution ' + str(self)
        alpha = 1.0 / z
        for e in self.support():
            self.mulProb(e, alpha)
        return self

def uniform_dist(elts):
    """
    Uniform distribution over a given finite set of C{elts}
    @param elts: list of any kind of item
    """
    p = 1.0 / len(elts)
    return DDist(dict([(e, p) for e in elts]))    

def delta_dist(elt):
    return DDist({elt: 1.0})

class mixture_dist(DDist):
    """
    A mixture of two probabability distributions, d1 and d2, with
    mixture parameter p.  Probability of an
    element x under this distribution is p * d1(x) + (1 - p) * d2(x).
    It is as if we first flip a probability-p coin to decide which
    distribution to draw from, and then choose from the approriate
    distribution.

    This implementation is lazy;  it stores the component
    distributions.  Alternatively, we could assume that d1 and d2 are
    DDists and compute a new DDist.
    """
    def __init__(self, d1, d2, p):
        self.d1 = d1
        self.d2 = d2
        self.p = p
        self.binom = DDist({True: p, False: 1 - p})
        
    def prob(self, elt):
        return self.p * self.d1.prob(elt) + (1 - self.p) * self.d2.prob(elt)

    def draw(self):
        if self.binom.draw():
            return self.d1.draw()
        else:
            return self.d2.draw()

    def support(self):
        return list(set(self.d1.support()).union(set(self.d2.support())))

    def __str__(self):
        result = 'MixtureDist({'
        elts = self.support()
        for x in elts[:-1]:
            result += str(x) + ' : ' + str(self.prob(x)) + ', '
        result += str(elts[-1]) + ' : ' + str(self.prob(elts[-1])) + '})'
        return result
    
    __repr__ = __str__


