import numpy as np
import matplotlib.pyplot as plt

def pca(x, varRetained = 0.95):
    ''' varRetained is the original data variance retained in the new dataset. New number of dimensions
     is computed based on the desired fraction of the original variance.
    '''
    n = x.shape[0]
    d = x.shape[1]
    sigma = 1.0/d*np.dot(x.T,x)
    U, S, V = np.linalg.svd(sigma, full_matrices=True)
    k = 0
    total_var = np.sum(S)
    var_cum_sums = np.array([np.sum(S[:i+1])/total_var*100 for i in range(d)])
    k = len(var_cum_sums[var_cum_sums<(varRetained*100)])
    U_reduced = U[:, : k]
    return np.dot(x, U_reduced), k

class FLDA:
    def fit(self, x, y):
        n = x.shape[0]
        x = np.concatenate([x, np.ones([n,1])], axis = 1)
        classes = set(y)
        n0= n1= 0
        for i in y:
            if(y==0):
                M0 += x[i]
                n0 += 1
            else:
                M1 += x[i]
                n1 += 1
        M0 = M0/n0
        M1 = M1/n1
        Sw = []
        for i in y:
            if(y==0):
                Sw += np.dot((x[i] - M0), (x[i] - M0).T)
            else:
                Sw += np.dot((x[i] - M1), (x[i] - M1).T)
        w = np.dot(np.linalg.inv(Sw), (M0- M1))
        return w
    def predict(self, x, w):
        h = np.dot(w, x)
        if(h>0):
            return 0
        else:
            return 1
