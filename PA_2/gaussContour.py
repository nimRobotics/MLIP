import matplotlib.pyplot as plt  # set plt as alias for matplotlib.pyplot
import numpy as np
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

from sympy import *
from sympy.solvers import solve
from sympy import Symbol

x1=Symbol('x1')
x2=Symbol('x2')

def plot2dGaussianCont(w1_mu,w1_sigma,w2_mu,w2_sigma,w3_mu,w3_sigma,eq1,eq2,eq3):
    #Create grid and multivariate normal
    x = np.linspace(-40,40,500)
    y = np.linspace(-40,40,500)
    X, Y = np.meshgrid(x,y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y

    rv1 = multivariate_normal(w1_mu,w1_sigma)
    rv2 = multivariate_normal(w2_mu,w2_sigma)
    rv3 = multivariate_normal(w3_mu,w3_sigma)

    plt.clf()
    plt.contour(X, Y, rv1.pdf(pos))
    plt.contour(X, Y, rv2.pdf(pos))
    plt.contour(X, Y, rv3.pdf(pos))
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Distribution of Data in feature space")
    plt.annotate('W1',(-5,-7))
    plt.annotate('W2',(3,15))
    plt.annotate('W3',(10,-12))

    # plotting the eigen vectors for w1_sigma
    evals, evecs = np.linalg.eig(w1_sigma)
    # print(evecs)
    eig_vec1 = evecs[:,0]
    eig_vec2 = evecs[:,1]
    print("Eigen vector 1 (w1_sigma): ",eig_vec1)
    print("Eigen vector 2 (w1_sigma): ",eig_vec2)
    plt.quiver(*w1_mu, *eig_vec1, color=['r'], scale=10)
    plt.quiver(*w1_mu, *eig_vec2, color=['b'], scale=10)

    # plotting the eigen vectors for w2_sigma
    evals, evecs = np.linalg.eig(w2_sigma)
    # print(evecs)
    eig_vec1 = evecs[:,0]
    eig_vec2 = evecs[:,1]
    print("Eigen vector 1 (w2_sigma): ",eig_vec1)
    print("Eigen vector 2 (w2_sigma): ",eig_vec2)
    plt.quiver(*w2_mu, *eig_vec1, color=['r'], scale=10)
    plt.quiver(*w2_mu, *eig_vec2, color=['b'], scale=10)

    # plotting the eigen vectors for w3_sigma
    evals, evecs = np.linalg.eig(w3_sigma)
    # print(evecs)
    eig_vec1 = evecs[:,0]
    eig_vec2 = evecs[:,1]
    print("Eigen vector 1 (w3_sigma): ",eig_vec1)
    print("Eigen vector 2 (w3_sigma): ",eig_vec2)
    plt.quiver(*w3_mu, *eig_vec1, color=['r'], scale=10)
    plt.quiver(*w3_mu, *eig_vec2, color=['b'], scale=10)

    # # Case 1
    # # print(eq1)
    # # print(solve(eq1,x2))
    # db1 = solve(eq1,x2)
    # lam_x = lambdify(x1, db1, modules=['numpy'])
    # x=np.linspace(-40,7.107,500)
    # y_vals = np.array(lam_x(x)).reshape((500,1))
    # plt.plot(x, y_vals[:],"black")
    #
    # db2 = solve(eq2,x2)
    # lam_x = lambdify(x1, db2, modules=['numpy'])
    # x=np.linspace(7.107,40,500)
    # y_vals = np.array(lam_x(x)).reshape((500,1))
    # plt.plot(x, y_vals[:],"black")
    #
    # db3 = solve(eq3,x2)
    # lam_x = lambdify(x1, db3, modules=['numpy'])
    # x=np.linspace(-40,7.107,500)
    # y_vals = np.array(lam_x(x)).reshape((500,1))
    # plt.plot(x, y_vals[:],"black")
    # # print(db1,db2,db3)
    # # print("intersection point",solve(db1[0]-db2[0],x1))

    ## case2
    # print(eq1)
    # print(solve(eq1,x2))
    # # print(x.shape)
    # db1 = solve(eq2,x2)
    # lam_x = lambdify(x1, db1, modules=['numpy'])
    # # x=np.linspace(-40,7.107,500)
    # y_vals = np.array(lam_x(x)).reshape((500,2))
    # plt.plot(x, y_vals[:,1],"black")

    plt.xlim((-40,40))
    plt.ylim((-40,40))
    plt.show()
