import numpy as np
import scipy
from functools import reduce
def normalize(x):
    y_max = 255
    y_min = 0
    x_max = np.max(np.max(x))
    x_min = np.min(np.min(x))
    out = np.around((y_max-y_min)*(x-x_min)/(x_max-x_min) + y_min)
    return out

# a = np.random.rand(5,5)
# print(normalize(a))

# class Data_Sparse():
#     def __int__(self):
#         self.max_iter = 100
#         self.tol = 1e-4
#         self.k = 3
#         self.lma = 1
def sparse_coding(x,k = 3,max_iter=100,lma=1,tol=1e-4):
    x = scipy.stats.zscore(x)
    n = x.shape[0]
    d = x.shape[1]
    x = x - np.mean(x)
    w = np.random.randn(d,k)
    for i in range(max_iter):
        alpha = np.dot(x,w)
        w_old = w
        for j in range(k):
            # w[:,j] = x.T * (x * w_old[:,j])/(w_old[:,j].T * x.T * x * w_old[:,j] + lma)
            w[:, j] = np.dot(x.T,np.dot(x,w_old[:, j]))/(reduce(np.dot, (w_old[:, j].T, x.T, x, w_old[:, j])) + lma)
            # w[:, j] =reduce(np.dot, (w_old[:, j].T, x.T, x, w_old[:, j])) + lma
        err = np.linalg.norm(x - np.dot(alpha,np.transpose(w)),'fro')
        if err < tol:
            break

    return alpha

b = [[1,2,3],[4,5,6],[7,8,9]]
a = sparse_coding(b,3,100,1,1e-4)
print(a)


