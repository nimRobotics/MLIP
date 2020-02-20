import matplotlib.pyplot as plt
import xlrd
import numpy as np
from sympy import *
from sympy.solvers import solve
from sympy import Symbol
from mvnpdf import plot2dGaussian
from gaussContour import plot2dGaussianCont

x1=Symbol('x1')
x2=Symbol('x2')

def mean_cov(dataFeatures):
    # find mean mu
    mu=(1/dataFeatures.shape[0])*dataFeatures.sum(axis=0)
    # find sigma
    xMINmu=dataFeatures-mu
    xMINmu_xMINmuTrans=np.zeros((2,2))
    for i in range(dataFeatures.shape[0]):
        xMINmu_xMINmuTrans = xMINmu_xMINmuTrans + np.dot(xMINmu[i,:].reshape((2,1)),np.transpose(xMINmu[i,:].reshape((2,1))))
    sigma=(1/dataFeatures.shape[0])*xMINmu_xMINmuTrans
    return mu,sigma

def decisionBoundary(w1_mu,w2_mu,w3_mu,w1_sigma,w2_sigma,w3_sigma):
    x=np.array([x1,x2])
    # class 1 & 2
    eq1=simplify(0.5*np.transpose(x-w1_mu).dot(np.linalg.inv(w1_sigma).dot(x-w1_mu))\
    -0.5*np.transpose(x-w2_mu).dot(np.linalg.inv(w2_sigma).dot(x-w2_mu))\
    +0.5*np.log(np.linalg.det(w1_sigma)) - 0.5*np.log(np.linalg.det(w2_sigma)))

    # class 2 & 3
    eq2=simplify(0.5*np.transpose(x-w2_mu).dot(np.linalg.inv(w2_sigma).dot(x-w2_mu))\
    -0.5*np.transpose(x-w3_mu).dot(np.linalg.inv(w3_sigma).dot(x-w3_mu))\
    +0.5*np.log(np.linalg.det(w2_sigma)) - 0.5*np.log(np.linalg.det(w3_sigma)))

    # class 3 & 1
    eq3=simplify(0.5*np.transpose(x-w3_mu).dot(np.linalg.inv(w3_sigma).dot(x-w3_mu))\
    -0.5*np.transpose(x-w1_mu).dot(np.linalg.inv(w1_sigma).dot(x-w1_mu))\
    +0.5*np.log(np.linalg.det(w3_sigma)) - 0.5*np.log(np.linalg.det(w1_sigma)))

    return eq1,eq2,eq3

def accuracy(eq1,eq2,eq3):
    truePrediction=0
    for i in range(w1_test.shape[0]):
        if eq1.subs(x1,w1_test[i][0]).subs(x2,w1_test[i][1])<0:
            truePrediction=truePrediction+1
    for i in range(w2_test.shape[0]):
        if eq2.subs(x1,w2_test[i][0]).subs(x2,w2_test[i][1])<0:
            truePrediction=truePrediction+1
    for i in range(w3_test.shape[0]):
        if eq3.subs(x1,w3_test[i][0]).subs(x2,w3_test[i][1])<0:
            truePrediction=truePrediction+1
    return truePrediction/(w1_test.shape[0]+w2_test.shape[0]+w3_test.shape[0])

def bayesian_classifier(case,w1_train,w1_test,w2_train,w2_test,w3_train,w3_test):
    #  Same covariance matrix for all the classes
    if case==1:
        _,sigma=mean_cov(np.concatenate((w1_train, w2_train, w3_train), axis=0))
        w1_mu,_=mean_cov(w1_train)
        w2_mu,_=mean_cov(w2_train)
        w3_mu,_=mean_cov(w3_train)

        print("Class w1 \n mu: ",w1_mu,"\n sigma: \n",sigma,"\n")
        print("Class w2 \n mu: ",w2_mu,"\n sigma: \n",sigma,"\n")
        print("Class w3 \n mu: ",w3_mu,"\n sigma: \n",sigma,"\n")

        eq1,eq2,eq3=decisionBoundary(w1_mu,w2_mu,w3_mu,sigma,sigma,sigma)
        acc=accuracy(eq1,eq2,eq3)
        print("Case 1 accuracy: ",acc)

        # plot2dGaussian(w1_mu,sigma,w2_mu,sigma,w3_mu,sigma)
        plot2dGaussianCont(w1_mu,sigma,w2_mu,sigma,w3_mu,sigma,eq1,eq2,eq3)

    # Different covariance matrices
    if case==2:
        w1_mu,w1_sigma=mean_cov(w1_train)
        w2_mu,w2_sigma=mean_cov(w2_train)
        w3_mu,w3_sigma=mean_cov(w3_train)
        print("Class w1 \n mu: ",w1_mu,"\n sigma: \n",w1_sigma,"\n")
        print("Class w2 \n mu: ",w2_mu,"\n sigma: \n",w2_sigma,"\n")
        print("Class w3 \n mu: ",w3_mu,"\n sigma: \n",w3_sigma,"\n")

        eq1,eq2,eq3=decisionBoundary(w1_mu,w2_mu,w3_mu,w1_sigma,w2_sigma,w3_sigma)
        acc=accuracy(eq1,eq2,eq3)
        print("Case 2 accuracy: ",acc)

        plot2dGaussian(w1_mu,w1_sigma,w2_mu,w2_sigma,w3_mu,w3_sigma)
        plot2dGaussianCont(w1_mu,w1_sigma,w2_mu,w2_sigma,w3_mu,w3_sigma,eq1,eq2,eq3)

    # Diagonal covariance matrices
    if case==3:

        w1_mu,w1_sigma=mean_cov(w1_train)
        w1_sigma=np.diag(np.diag(w1_sigma))

        w2_mu,w2_sigma=mean_cov(w2_train)
        w2_sigma=np.diag(np.diag(w2_sigma))

        w3_mu,w3_sigma=mean_cov(w3_train)
        w3_sigma=np.diag(np.diag(w3_sigma))

        print("Class w1 \n mu: ",w1_mu,"\n sigma: \n",w1_sigma,"\n")
        print("Class w2 \n mu: ",w2_mu,"\n sigma: \n",w2_sigma,"\n")
        print("Class w3 \n mu: ",w3_mu,"\n sigma: \n",w3_sigma,"\n")

        eq1,eq2,eq3=decisionBoundary(w1_mu,w2_mu,w3_mu,w1_sigma,w2_sigma,w3_sigma)
        acc=accuracy(eq1,eq2,eq3)
        print("Case 3 accuracy: ",acc)

        plot2dGaussian(w1_mu,w1_sigma,w2_mu,w2_sigma,w3_mu,w3_sigma)
        plot2dGaussianCont(w1_mu,w1_sigma,w2_mu,w2_sigma,w3_mu,w3_sigma,eq1,eq2,eq3)

# loading the data
loc = ("que3.xlsx")
wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)

data=[]
for i in range(1,1501):
    data.append([sheet.cell_value(i, 0),sheet.cell_value(i, 1)])

# list to np array
data = np.array(data)

# splitting data between 3 classes
w1=data[:500,:]
w2=data[501:1000,:]
w3=data[1001:1500,:]

# splitting test and train data
w1_train=w1[:350,:]
w1_test=w1[351:,:]
w2_train=w2[:350,:]
w2_test=w2[351:,:]
w3_train=w3[:350,:]
w3_test=w3[351:,:]

# the first parameter describes the case and can be 1/2/3
bayesian_classifier(3,w1_train,w1_test,w2_train,w2_test,w3_train,w3_test)
