import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from numpy import linalg as LA

# Assume that I loaded 'N' no of 2d points from a file and used
# np.cov() to find the below covariance matrix

# This is my covariance matrix obtained from 2 x N points
cov_mat = [[3407.3108669,  1473.06388943],
           [1473.06388943, 1169.53151003]]

eigen_values, eigen_vectors = LA.eig(cov_mat)

origin = [0, 0]

# print(eigen_vectors)
eig_vec1 = eigen_vectors[:,0]
eig_vec2 = eigen_vectors[:,1]
print("Eigen vector 1: ",eig_vec1)
print("Eigen vector 2: ",eig_vec2)

plt.quiver(*origin, *eig_vec1, color=['r'], scale=21)
plt.quiver(*origin, *eig_vec2, color=['b'], scale=21)
plt.show()
