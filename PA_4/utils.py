import matplotlib.pyplot as plt
from random import randint

def visualise(x_train):
    for i in range(64):
        ax = plt.subplot(8, 8, i+1)
        ax.axis('off')
        plt.imshow(x_train[randint(0, x_train.shape[0])].reshape(28,28), cmap='Greys')
    plt.show()
