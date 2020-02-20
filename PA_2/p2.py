import matplotlib.pyplot as plt
import xlrd
import numpy as np
from sympy import *
from sympy.solvers import solve
from sympy import Symbol


# load data with kth feature
def load_data(k):
    # reading the xls file
    loc = ("ImageSegData.xls")
    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)

    brickFeat=[]
    grassFeat=[]
    pathFeat=[]
    skyFeat=[]
    for i in range(210):
        if sheet.cell_value(i, 0)=="BRICKFACE":
            brickFeat.append(sheet.cell_value(i, k))
        if sheet.cell_value(i, 0)=="PATH":
            pathFeat.append(sheet.cell_value(i, k))
        if sheet.cell_value(i, 0)=="GRASS":
            grassFeat.append(sheet.cell_value(i, k))
        if sheet.cell_value(i, 0)=="SKY":
            skyFeat.append(sheet.cell_value(i, k))

    # train set
    w1_train=np.array(brickFeat)
    w2_train=np.array(grassFeat)
    w3_train=np.array(pathFeat)

    # loading the test set
    loc = ("ImageSegData_test.xls")
    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)
    brickTest=[]
    grassTest=[]
    pathTest=[]
    for i in range(210):
        if sheet.cell_value(i, 0)=="BRICKFACE":
            brickTest.append(sheet.cell_value(i, 1))
        if sheet.cell_value(i, 0)=="PATH":
            grassTest.append(sheet.cell_value(i, 1))
        if sheet.cell_value(i, 0)=="GRASS":
            pathTest.append(sheet.cell_value(i, 1))

    w1_test=np.array(brickTest)
    w2_test=np.array(grassTest)
    w3_test=np.array(pathTest)
    return(w1_train,w2_train,w3_train,w1_test,w2_test,w3_test)

# plotting the Histogram
def plot_hist():
    bins=10
    plt.subplot(221)
    n, bins, patches = plt.hist(brickFeat, bins, color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('region-centroid-col (Class:Brickface)')
    plt.ylabel('Frequency')

    plt.subplot(222)
    n, bins, patches = plt.hist(pathFeat, bins, color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('region-centroid-col (Class:Path)')
    plt.ylabel('Frequency')

    plt.subplot(223)
    n, bins, patches = plt.hist(grassFeat, bins, color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('region-centroid-col (Class:Grass)')
    plt.ylabel('Frequency')

    plt.subplot(224)
    n, bins, patches = plt.hist(skyFeat, bins, color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('region-centroid-col (Class:Sky)')
    plt.ylabel('Frequency')
    plt.suptitle("1D Histogram for 4 classes")
    plt.show()

# mean cov matric for single feature
def mean_cov_sf(dataFeatures):
    # find mean mu
    mu=(1/dataFeatures.shape[0])*dataFeatures.sum(axis=0)
    # find sigma
    xMINmu=dataFeatures-mu
    sigma=(1/dataFeatures.shape[0])*xMINmu.dot(xMINmu.T)
    return mu,sigma

# db for single feature
def decisionBoundary(w1_mu,w2_mu,w3_mu,w1_sigma,w2_sigma,w3_sigma):
    # class 1 & 2
    eq1=-((x-w2_mu)**2)/w2_sigma - np.log(w2_sigma) + ((x-w1_mu)**2)/w1_sigma + np.log(w1_sigma)

    # class 2 & 3
    eq2=-((x-w3_mu)**2)/w3_sigma - np.log(w3_sigma) + ((x-w2_mu)**2)/w2_sigma + np.log(w2_sigma)

    # class 3 & 1
    eq3=-((x-w1_mu)**2)/w1_sigma - np.log(w1_sigma) + ((x-w3_mu)**2)/w3_sigma + np.log(w3_sigma)

    return eq1,eq2,eq3

# finding accuracy on test data
def accuracy(eq1,eq2,eq3):
    truePrediction=0
    for i in range(w1_test.shape[0]):
        if eq1.subs(x,w1_test[i])<0:
            truePrediction=truePrediction+1
    for i in range(w2_test.shape[0]):
        if eq2.subs(x,w2_test[i])<0:
            truePrediction=truePrediction+1
    for i in range(w3_test.shape[0]):
        if eq3.subs(x,w3_test[i])<0:
            truePrediction=truePrediction+1
    return truePrediction/(w1_test.shape[0]+w2_test.shape[0]+w3_test.shape[0])


featureCols=[1,2,4,6,8,10,11,12,13,14,15]
accuracies=[]
for e in featureCols:
    w1_train,w2_train,w3_train,w1_test,w2_test,w3_test=load_data(e)
    x=Symbol('x')

    w1_mu,w1_sigma=mean_cov_sf(w1_train)
    w2_mu,w2_sigma=mean_cov_sf(w2_train)
    w3_mu,w3_sigma=mean_cov_sf(w3_train)

    print("Class w1 \n mu: ",w1_mu,"\n sigma: \n",w1_sigma,"\n")
    print("Class w2 \n mu: ",w2_mu,"\n sigma: \n",w2_sigma,"\n")
    print("Class w3 \n mu: ",w3_mu,"\n sigma: \n",w3_sigma,"\n")

    eq1,eq2,eq3=decisionBoundary(w1_mu,w2_mu,w3_mu,w1_sigma,w2_sigma,w3_sigma)
    # print(eq1,eq2,eq3)
    acc=accuracy(eq1,eq2,eq3)
    accuracies.append(acc)
    print("Accuracy (feature col",e,"): ",acc)
# print(accuracies)
plt.scatter(featureCols,accuracies)
plt.xlabel("feature column number")
plt.ylabel("accuracy")
plt.title("Classification with single feature for 3 classes (brick,grass,path)")
plt.show()
