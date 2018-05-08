import numpy as np
import modules_disp as disp
from math import log
from math import e
class Module:
    def sgd_step(self, lrate): pass     # For modules w/o weights

class Linear(Module):
    def __init__(self,m,n):
        self.m, self.n = (m, n)         # (in size, out size)
        self.W0 = np.zeros([self.n,1])  # (n x 1)
        self.W = np.random.normal(0,1.0*m**(-.5),[m,n]) # (m x n)
    def forward(self,A):
        self.A = A                      # (m x b)
        return None                     # Your code (n x b)
    def backward(self,dLdZ):            # dLdZ is (n x b), uses stored self.A
        self.dLdW = None                  # Your code
        self.dLdW0 = None                 # Your code
        return None                     # Your code (m x b)
    def sgd_step(self, lrate):          # Gradient descent step
        self.W = None                   # Your code
        self.W0 = None                  # Your code

class Tanh(Module):                     # Layer activation
    def forward(self,Z):
        self.A = np.tanh(Z)
        return self.A
    def backward(self,dLdA):            # Uses stored self.A
        return None                     # Your code

class ReLU(Module):                     # Layer activation
    def forward(self,Z):
        self.A = None                   # Your code
        return self.A
    def backward(self,dLdA):		# uses stored self.A
        return None                     # Your code

class SoftMax(Module):                  # Output activation
    def forward(self,Z):
        return None                     # Your code
    def backward(self,dLdZ):            # Assume that dLdZ is passed in
        return dLdZ
    def class_fun(self, Ypred):         # Return class indices
        return None                     # Your code

class NLL(Module):                      # Loss
    def forward(self,Ypred,Y):
        self.Ypred = Ypred
        self.Y = Y
        return None                     # Your code
    def backward(self):                 # Use stored self.Ypred, self.Y
        return None                     # Your code (see end of 5.2)

class Sequential:
    def __init__(self, modules, loss):  # List of modules, loss module
        self.modules = modules
        self.loss = loss
    def sgd(self, X, Y, iters = 100, lrate = 0.005): # Train
        D,N = X.shape
        for it in range(iters):
            pass                        # Your code
    def forward(self,X):                # Compute Ypred
        for m in self.modules: X = m.forward(X)
        return X
    def backward(self,delta):              # Update dLdW and dLdW0
        # Note reversered list of modules
        for m in self.modules[::-1]: delta = m.backward(delta)
    def sgd_step(self,lrate):           # Gradient descent step
        for m in self.modules: m.sgd_step(lrate)

######################################################################
#   Data Sets
######################################################################

def super_simple_separable_through_origin():
    X = np.array([[2, 3, 9, 12],
                  [5, 1, 6, 5]])
    y = np.array([[1, 0, 1, 0]])
    return X, for_softmax(y)

def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, 0, 1, 0]])
    return X, for_softmax(y)

def xor():
    X = np.array([[1, 2, 1, 2],
                  [1, 2, 2, 1]])
    y = np.array([[1, 1, 0, 0]])
    return X, for_softmax(y)

def xor_more():
    X = np.array([[1, 2, 1, 2, 2, 4, 1, 3],
                  [1, 2, 2, 1, 3, 1, 3, 3]])
    y = np.array([[1, 1, 0, 0, 1, 1, 0, 0]])
    return X, for_softmax(y)

def hard():
    X= np.array([[-0.23390341,  1.18151883, -2.46493986,  1.55322202,  1.27621763,
              2.39710997, -1.3440304 , -0.46903436, -0.64673502, -1.44029872,
              -1.37537243,  1.05994811, -0.93311512,  1.02735575, -0.84138778,
              -2.22585412, -0.42591102,  1.03561105,  0.91125595, -2.26550369],
             [-0.92254932, -1.1030963 , -2.41956036, -1.15509002, -1.04805327,
              0.08717325,  0.8184725 , -0.75171045,  0.60664705,  0.80410947,
              -0.11600488,  1.03747218, -0.67210575,  0.99944446, -0.65559838,
              -0.40744784, -0.58367642,  1.0597278 , -0.95991874, -1.41720255]])
    y= np.array([[ 1.,  1., 0.,  1.,  1.,  1., 0., 0., 0., 0., 0.,  1.,  1.,
                   1., 0., 0., 0.,  1.,  1., 0.]])
    return X, for_softmax(y)

def for_softmax(y):
    return np.vstack([y, 1-y])

######################################################################
# Tests
######################################################################

def sgd_test(nn):
    lrate = 0.005
    # data
    X,Y = super_simple_separable()
    print('X\n', X)
    print('Y\n', Y)
    # define the modules
    assert len(nn.modules) == 4
    (l_1, f_1, l_2, f_2) = nn.modules
    Loss = nn.loss

    print('l_1.W\n', l_1.W)
    print('l_1.W0\n', l_1.W0)
    print('l_2.W\n', l_2.W)
    print('l_2.W0\n', l_2.W0)

    z_1 = l_1.forward(X)
    print('z_1\n', z_1)
    a_1 = f_1.forward(z_1)
    print('a_1\n', a_1)
    z_2 = l_2.forward(a_1)
    print('z_2\n', z_2)
    a_2 = f_2.forward(z_2)
    print('a_2\n', a_2)

    Ypred = a_2
    loss = Loss.forward(Ypred, Y)
    print('loss\n', loss)
    dloss = Loss.backward()
    print('dloss\n', dloss)

    dL_dz2 = f_2.backward(dloss)
    print('dL_dz2\n', dL_dz2)
    dL_da1 = l_2.backward(dL_dz2)
    print('dL_da1\n', dL_da1)
    dL_dz1 = f_1.backward(dL_da1)
    print('dL_dz1\n', dL_dz1)
    dL_dX = l_1.backward(dL_dz1)
    print('dL_dX\n', dL_dX)

    l_1.sgd_step(lrate)
    print('l_1.W\n', l_1.W)
    print('l_1.W0\n', l_1.W0)
    l_2.sgd_step(lrate)
    print('l_2.W\n', l_2.W)
    print('l_2.W0\n', l_2.W0)

######################################################################
# Desired output
######################################################################

'''
# sgd_test for Tanh activation and SoftMax output
# np.random.seed(0)
# sgd_test(Sequential([Linear(2,3), Tanh(), Linear(3,2), SoftMax()], NLL()))
X
 [[ 2  3  9 12]
  [ 5  2  6  5]]
Y
 [[1 0 1 0]
  [0 1 0 1]]
l_1.W
 [[ 1.24737338  0.28295388  0.69207227]
  [ 1.58455078  1.32056292 -0.69103982]]
l_1.W0
 [[ 0.]
  [ 0.]
  [ 0.]]
l_2.W
 [[ 0.5485338  -0.08738612]
  [-0.05959343  0.23705916]
  [ 0.08316359  0.8396252 ]]
l_2.W0
 [[ 0.]
  [ 0.]]
z_1
 [[ 10.41750064   6.91122168  20.73366505  22.8912344 ]
  [  7.16872235   3.48998746  10.46996239   9.9982611 ]
  [ -2.07105455   0.69413716   2.08241149   4.84966811]]
a_1
 [[ 1.          0.99999801  1.          1.        ]
  [ 0.99999881  0.99814108  1.          1.        ]
  [-0.96871843  0.60063321  0.96941021  0.99987736]]
z_2
 [[ 0.40837833  0.53900088  0.56956001  0.57209377]
  [-0.66368766  0.65353931  0.96361427  0.98919526]]
a_2
 [[ 0.74498961  0.47139666  0.4027417   0.39721055]
  [ 0.25501039  0.52860334  0.5972583   0.60278945]]
loss
 2.3475491206369514
dloss
 [[-0.25501039  0.47139666 -0.5972583   0.39721055]
  [ 0.25501039 -0.47139666  0.5972583  -0.39721055]]
dL_dz2
 [[-0.25501039  0.47139666 -0.5972583   0.39721055]
  [ 0.25501039 -0.47139666  0.5972583  -0.39721055]]
dL_da1
 [[-0.16216619  0.29977053 -0.37980845  0.2525941 ]
  [ 0.07564949 -0.13984104  0.17717822 -0.11783354]
  [ 0.19290557 -0.35659347  0.45180297 -0.30047453]]
dL_dz1
 [[ -5.80088442e-10   1.19079549e-06  -0.00000000e+00   0.00000000e+00]
  [  1.79552879e-07  -5.19424389e-04   5.70658808e-10  -9.74876621e-10]
  [  1.18800113e-02  -2.27948719e-01   2.72183509e-02  -7.36963862e-05]]
dL_dX
 [[  8.22187641e-03  -1.57902474e-01   1.88370660e-02  -5.10035008e-05]
  [ -8.20932462e-03   1.56837595e-01  -1.88089635e-02   5.09258498e-05]]
l_1.W
 [[ 1.24737336  0.28296167  0.69415229]
  [ 1.58455077  1.32056811 -0.68987204]]
l_1.W0
 [[ -5.95107701e-09]
  [  2.59622620e-06]
  [  9.44620265e-04]]
l_2.W
 [[ 0.54845212 -0.08730444]
  [-0.05967074  0.23713647]
  [ 0.08142188  0.84136692]]
l_2.W0
[[ -8.16925787e-05]
 [  8.16925787e-05]]
'''

'''
# Compare to the results above
np.random.seed(0)
nn = Sequential([Linear(2, 3), Tanh(), Linear(3,2), SoftMax()], NLL())
# These should match the initial weights above
print('-----------------')
print(nn.modules[0].W, '\n', nn.modules[0].W0)
print(nn.modules[2].W, '\n', nn.modules[2].W0)
# Run one iteration
nn.sgd(X,Y, iters = 1, lrate=lrate)
# These should match the final weights above
print('-----------------')
print(nn.modules[0].W, '\n', nn.modules[0].W0)
print(nn.modules[2].W, '\n', nn.modules[2].W0)
'''

'''
# sgd_test for ReLU activation and SoftMax output
# np.random.seed(0)
# sgd_test(Sequential([Linear(2,3), ReLU(), Linear(3,2), SoftMax()], NLL()))
X
 [[ 2  3  9 12]
  [ 5  2  6  5]]
Y
 [[1 0 1 0]
  [0 1 0 1]]
l_1.W
 [[ 1.24737338  0.28295388  0.69207227]
  [ 1.58455078  1.32056292 -0.69103982]]
l_1.W0
 [[ 0.]
  [ 0.]
  [ 0.]]
l_2.W
 [[ 0.5485338  -0.08738612]
  [-0.05959343  0.23705916]
  [ 0.08316359  0.8396252 ]]
l_2.W0
 [[ 0.]
  [ 0.]]
z_1
 [[ 10.41750064   6.91122168  20.73366505  22.8912344 ]
  [  7.16872235   3.48998746  10.46996239   9.9982611 ]
  [ -2.07105455   0.69413716   2.08241149   4.84966811]]
a_1
 [[ 10.41750064   6.91122168  20.73366505  22.8912344 ]
  [  7.16872235   3.48998746  10.46996239   9.9982611 ]
  [  0.           0.69413716   2.08241149   4.84966811]]
z_2
 [[  5.28714248   3.64078533  10.92235599  12.36410102]
  [  0.78906625   0.80620366   2.41861097   4.44170662]]
a_2
 [[  9.88992134e-01   9.44516196e-01   9.99797333e-01   9.99637598e-01]
  [  1.10078665e-02   5.54838042e-02   2.02666719e-04   3.62401857e-04]]
loss
 10.8256925657554
dloss
 [[ -1.10078665e-02   9.44516196e-01  -2.02666719e-04   9.99637598e-01]
  [  1.10078665e-02  -9.44516196e-01   2.02666719e-04  -9.99637598e-01]]
dL_dz2
 [[ -1.10078665e-02   9.44516196e-01  -2.02666719e-04   9.99637598e-01]
  [  1.10078665e-02  -9.44516196e-01   2.02666719e-04  -9.99637598e-01]]
dL_da1
 [[ -7.00012165e-03   6.00636672e-01  -1.28879806e-04   6.35689470e-01]
  [  3.26551207e-03  -2.80193173e-01   6.01216067e-05  -2.96545080e-01]
  [  8.32702834e-03  -7.14490239e-01   1.53309592e-04  -7.56187463e-01]]
dL_dz1
 [[ -7.00012165e-03   6.00636672e-01  -1.28879806e-04   6.35689470e-01]
  [  3.26551207e-03  -2.80193173e-01   6.01216067e-05  -2.96545080e-01]
  [  0.00000000e+00  -7.14490239e-01   1.53309592e-04  -7.56187463e-01]]
dL_dX
 [[ -7.80777608e-03   1.75457571e-01  -3.76482800e-05   1.85697170e-01]
  [ -6.77973405e-03   1.07546779e+00  -2.30765264e-04   1.13823145e+00]]
l_1.W
 [[ 1.20029826  0.30491412  0.74815397]
  [ 1.56283104  1.33069504 -0.66499483]]
l_1.W0
 [[-0.00614599]
  [ 0.00286706]
  [ 0.00735262]]
l_2.W
 [[ 0.40207469  0.05907299]
  [-0.1256432   0.30310892]
  [ 0.05564803  0.86714076]]
l_2.W0
 [[-0.00966472]
  [ 0.00966472]]
 '''

'''
X, Y = hard()
nn = Sequential([Linear(2, 10), ReLU(), Linear(10, 10), ReLU(), Linear(10,2), SoftMax()], NLL())
disp.classify(X, Y, nn, it=100000)
'''

#######
# Test cases
######

def nn_tanh_test():
    np.random.seed(0)
    nn = Sequential([Linear(2,3), Tanh(), Linear(3,2), SoftMax()], NLL())
    X,Y = super_simple_separable()
    nn.sgd(X,Y, iters = 1, lrate=0.005)
    return [np.vstack([nn.modules[0].W, nn.modules[0].W0.T]).tolist(),
            np.vstack([nn.modules[2].W, nn.modules[2].W0.T]).tolist()]

def nn_relu_test():
    np.random.seed(0)
    nn = Sequential([Linear(2,3), ReLU(), Linear(3,2), SoftMax()], NLL())
    X,Y = super_simple_separable()
    nn.sgd(X,Y, iters = 2, lrate=0.005)
    return [np.vstack([nn.modules[0].W, nn.modules[0].W0.T]).tolist(),
            np.vstack([nn.modules[2].W, nn.modules[2].W0.T]).tolist()]

def nn_pred_test():
    np.random.seed(0)
    nn = Sequential([Linear(2,3), ReLU(), Linear(3,2), SoftMax()], NLL())
    X,Y = super_simple_separable()
    nn.sgd(X,Y, iters = 1, lrate=0.005)
    Ypred = nn.forward(X)
    return nn.modules[-1].class_fun(Ypred).tolist(), [nn.loss.forward(Ypred, Y)]

def NLL(y, a):
    n, _ = y.shape
    sum = 0
    for i in range(n):
        sum -= y[i]*log(a[i])
    return sum

def SM(z):
    n, _ = z.shape
    denom = 0
    for i in range(n):
        denom += e**z[i]
    a = np.zeros(z.shape)
    for i in range(n):
        a[i] = e**z[i]/denom
    return a

def grad_NLL_W(a, y, x):
    d, _ = x.shape
    n, _ = a.shape
    dW = np.dot(x, (a-y).T)
    return dW

def ReLU(x):
    return x if x > 0 else 0

def FC(W, x, W_0):
    return np.dot(W.T, x) + W_0


if __name__ == "__main__":
    # y = np.array([[0,0,1]])
    # a = np.array([[.3,.5,.2]])
    # W = np.array([[1, -1, -2],
    #               [-1, 2, 1]])
    # x = np.array([[1,1]]).T
    # y = np.array([[0, 1, 0]]).T
    # # z = np.array([[-1, 0, 1]]).T
    # # print(SM(z))
    # # print(NLL(y.T, a.T))
    # z = np.dot(W.T, x)
    # a = SM(z)
    # # print(a)
    # # print(grad_NLL_W(a, y, x).tolist())
    # dW = grad_NLL_W(a, y, x)
    # W = W - .5 * dW
    # # print(W.tolist())
    # z = np.dot(W.T, x)
    # a = SM(z)
    # print(a)
    W = np.array([[1, 0, -1, 0],
                   [0, 1, 0, -1]])
    W_0 = np.array([[-1, -1, -1, -1]]).T
    V = np.array([[1, -1],
                  [1, -1],
                  [1, -1],
                  [1, -1]])
    V_0 = np.array([[0, 2]]).T
    x = np.array([[3, 14]]).T
    z = FC(W, x, W_0)
    ReLU_np = np.vectorize(ReLU)
    f_z = ReLU_np(z)
    print('f_z: ', f_z.tolist())
    u = FC(V, f_z, V_0)
    print('u: ', u)
    f_u = ReLU_np(u)
    print('f_u: ', f_u)
    print('SM: ', SM(f_u))
    X = np.array([[.5, 0, -3], [.5, 2, .5]])
    print(ReLU_np(FC(W,X, W_0)).tolist())