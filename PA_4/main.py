import numpy as np
from utils import visualise
from read_mnist import load_data
import random

y_train,x_train,y_test,x_test=load_data()
print("Train data label dim: {}".format(y_train.shape))
print("Train data features dim: {}".format(x_train.shape))
print("Test data label dim: {}".format(y_test.shape))
print("Test data features dim:{}".format(x_test.shape))

# uncomment to visualise dataset
# visualise(x_train)

def sigmoid(x):
    return 1/(1+ np.exp(-x))

def sigmoid_grad(x):
    return sigmoid(x).T @ (1 - sigmoid(x))

def softmax(x):
    for i,f in enumerate(x):
        f -= np.max(f) # for numerical stabiluty
        p = np.exp(f) / np.sum(np.exp(f))
        x[i,:]=p
    return x

def cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    	Note that y is not one-hot encoded vector.
    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    m = y.shape[0]
    p = softmax(X)
    log_likelihood = -np.log(p[range(m),y])
    loss = np.sum(log_likelihood) / m
    return loss

# https://deepnotes.io/softmax-crossentropy
def delta_cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    	Note that y is not one-hot encoded vector.
    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    m = y.shape[0]
    grad = softmax(X)
    grad[range(m),y] -= 1
    grad = grad/m
    return grad

class NN(object):
    def __init__(self, hidden_layers, hidden_neurons, hidden_activation, output_activation):
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.step_size=0.05

        self.W1 = 0.01* np.random.randn(x_train.shape[1],self.hidden_neurons)
        self.b1 = np.zeros((1,self.hidden_neurons))
        self.W2 = 0.01* np.random.randn(self.hidden_neurons,10)
        self.b2 = np.zeros((1,10))

    def forward(self,x_train):

        # print(self.W.shape,x_train.shape)
        score1=np.dot(x_train, self.W1) + self.b1
        # print("score1 dims: ", score1.shape)

        y = (sigmoid(score1))
        # print("y dims: ",y.shape)

        score2 = np.dot(y, self.W2) + self.b2
        # print("score2 dims: ", score2.shape)

        z = softmax(score2)
        # print("z (softmax) dims: ",z.shape)

        loss=cross_entropy(score2,y_train)
        # print("Loss",loss)

        return(loss,score2,y,z,score1)


    def backward(self):
        "J: cross entropy loss"
        djdscore2=delta_cross_entropy(score2,y_train)

        # updating w2,b2
        dW2 = np.dot(y.T, djdscore2)
        db2 = np.sum(djdscore2, axis=0)
        self.W2 += -self.step_size * dW2
        self.b2 += -self.step_size * db2

        # print("djdscore2 dims: ",djdscore2.shape)
        # print("w2 dims: ",self.W2.shape)
        # print("dw2 dims: ",dW2.shape)
        # print("db2 dims: ",db2.shape)


        # updating w1,b1
        dW1 = np.dot(x_train.T, np.dot(np.dot(djdscore2,self.W2.T),sigmoid_grad(score1)))
        db1 = np.sum(np.dot(np.dot(djdscore2,self.W2.T),sigmoid_grad(score1)),axis=0)
        db1.reshape(1,256)
        self.W1 += -self.step_size * dW1
        self.b1 += -self.step_size * db1
        # print("dW1:", dW1, "db1", db1, "dW2:", dW2, "db2", db2)
        # print("sigmoid_grad score1 dims: ",sigmoid_grad(score1).shape)
        # print("w1 dims: ",self.W1.shape)
        # print("b1 dims: ",self.b1.shape)
        # print("dw1 dims: ",dW1.shape)
        # print("db1 dims: ",db1.shape)



# def preprocess(X):
#     # zero center the data
#     X -= np.mean(X, axis = 0)
#     return X
#
# # preprocessing the image
# x_train=preprocess(x_train)
# x_test=preprocess(x_test)

model=NN(5,256,"sigmoid","softmax")

epochs=10
for epoch in range(epochs):
    loss,score2,y,z,score1 = model.forward(x_train)
    print("Loss: {} in {}/{}".format(loss,epoch,epochs))
    model.backward()
print(z.shape)
preds= np.argmax(z, axis=1)
print(preds.shape)
# print('training accuracy: %.2f' % (np.mean(preds == y_train)))
print(preds)
