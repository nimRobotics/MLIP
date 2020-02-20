# https://stackoverflow.com/questions/38698277/plot-normal-distribution-in-3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

def plot2dGaussian(w1_mu,w1_sigma,w2_mu,w2_sigma,w3_mu,w3_sigma):
    #Create grid and multivariate normal
    x = np.linspace(-20,20,500)
    y = np.linspace(-20,20,500)
    X, Y = np.meshgrid(x,y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y
    rv = multivariate_normal(w1_mu,w1_sigma)
    rv2 = multivariate_normal(w2_mu,w2_sigma)
    rv3 = multivariate_normal(w3_mu,w3_sigma)

    #Make a 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, rv.pdf(pos), rstride=3, cstride=3, cmap='viridis',linewidth=5)
    ax.plot_surface(X, Y, rv2.pdf(pos), rstride=3, cstride=3, cmap='viridis',linewidth=5)
    ax.plot_surface(X, Y, rv3.pdf(pos), rstride=3, cstride=3, cmap='viridis',linewidth=5)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Z axis')
    plt.show()
