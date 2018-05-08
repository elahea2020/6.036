import numpy as np

def fc(X, W, b):
    return np.dot(np.transpose(W), X) + b

if __name__ == "__main__":
    X = np.array([[0, 1, 2],
                  [0, 1, 2]])
    Y = np.array([[0, 1, 0]])
    W = np.array([[1, -1], [0, 0]])
    W0 = np.array([[-.5],[1.5]])
    # print(X.shape)
    # print(Y.shape)
    # print(W.shape)
    out = fc(X, W, W0)
    # print(np.sign(out).tolist())
    # v_func = np.vectorize(lambda x: 1 if x > 0 else 0)
    # print(v_func(out))
    # V = np.array([[2/3],[-2/3]])
    # print(out)
    # V_0 = np.array([[1]])
    # out_2 = fc(out, V, V_0)
    # print(out_2)
    # print(v_func(out_2))
    V = [[0],[0]]
    V_0 = [[-1]]
    x = [[1],[2]]
    print(fc(x, V, V_0))

