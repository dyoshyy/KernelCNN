from matplotlib import pyplot as plt
from matplotlib import offsetbox
import numpy as np
import sys
import math
import scipy.linalg
from numpy import linalg as LA
from sklearn.metrics import pairwise

from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh, lobpcg
from scipy.sparse.csgraph import connected_components

from sklearn import datasets
from sklearn.utils import check_random_state, check_array, check_symmetric
from sklearn.base import BaseEstimator
from sklearn.utils.extmath import _deterministic_vector_sign_flip
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestCentroid

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomTreesEmbedding

from scipy.sparse import csr_matrix, isspmatrix
from matplotlib.colors import LinearSegmentedColormap    #
import random
from sklearn.metrics import classification_report, accuracy_score

class KSLE_ML():
    def __init__(self, n_components=2, affinity="nearest_neighbors",
                 random_state=None, eigen_solver=None,
                 n_neighbors=None, kernel=None, kopt1=None, n_jobs=1, la=None):
        self.n_components = n_components
        self.affinity = affinity
        self.random_state = random_state
        self.eigen_solver = eigen_solver
        self.n_neighbors = n_neighbors
        #print(n_neighbors)
        self.n_jobs = n_jobs
        self.la = la
 
        # Kernel selection with option
        self.kernel = kernel
        self.kopt1 = kopt1

    @property
    def _pairwise(self):
        return self.affinity == "precomputed"

    def _change_kernel(self, kernel, kopt1=None, setonly =False):
        self.kernel = kernel
        self.kopt1 = kopt1
        ans = True
        if setonly == False:
            G = self._calc_kernel(self.X, self.X)
            self.G = G

            # Solve linear equation GC = Z instead of C = G^{-1}Z 
            rank = np.linalg.matrix_rank(G)
            print('Rank of G is '+str(rank))

            try:
                self.C = np.linalg.solve(G, self.Z)
            except np.linalg.LinAlgError as e:
                if 'Singular matrix' in str(e):
                        exit()
            ans = (rank==self.X.shape[0])
            print('(change)self.C.shape = '+str(self.C.shape)+' rank test='+str(ans))
        return ans

    def _calc_kernel(self, X, Y=None):
        print('Used kernel = ' + self.kernel + " w opt1 = "+ str(self.kopt1))
        if self.kernel == "polynomial":
            k  = pairwise.polynomial_kernel(X, Y, degree=self.kopt1)
        elif self.kernel == "rbf":
            k  = pairwise.rbf_kernel(X, Y, gamma=self.kopt1)
        elif self.kernel == "sinc":
            k  = sinc_kernel(X, Y, alpha=self.kopt1)

        if X.shape == Y.shape:
            print ('Gram matrix was (re)calculated.')
            self.G = k
        return k

    def explicit_map(self,Y):
        C = self.C                           # coefficients of kernels
        k = self._calc_kernel(self.X,Y)      # innerproduct to the training sample via kernel
        Z =  np.dot(C.T,k)
        return  Z

    def _get_affinity_matrix(self, X, Y=None):
        self.X = X
        length=X.shape[0]
        la = self.la
      
        n = self.n_neighbors
        dist = np.zeros(length*length).reshape((length,length))

        N = np.zeros(length*length).reshape((length,length))
        nn = NearestNeighbors(n_neighbors=n+1)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)
        self.distances = distances
        self.indices = indices

        for i in range(length):
            for j in indices[i][1:n]:  # indices[i][0]=i itself
                N[i][j] = 1            # 1 if j is i's one of k NN's

        w_F = weight = (N + N.T)*0.5 #return 0,1 or 0.5

        if Y.ndim == 1:             # Y[i]= j \in {1,2,...,C}
            for i in range(length):
                for j in range(length):
                    w_L = 1.0 if (Y[i]==Y[j]) else 0.0

                    weight[i][j] = la*w_F[i][j] + (1-la)*w_L
                    if i==j:
                        weight[i][j]=0.0
        else:                      # Y[i] = (0,1,0,...1)
            for i in range(length):
                for j in range(length):

                    A = np.sum(np.logical_and(Y[i], Y[j])) 
                    B = np.sum(np.logical_or(Y[i], Y[j]))
                    w_L = float(A)/float(B)

                    weight[i][j] = la*w_F[i][j] + (1-la)*w_L
                    if i==j:
                        weight[i][j]=0.0

        self.affinity_matrix_ = weight
        return self.affinity_matrix_


    def get_parameter(self):  # depends on the kernel to be used
        distances = self.distances
        indices = self.indices
        n = self.X

        # In case of RBF kernel, estimate a variance parameter from data
        alpha = np.sum (np.sum(np.power(distances,2), axis=1)/self.n_neighbors) / self.X.shape[0]
        return alpha



    def fit(self, X,  y=None, map_d=2,):
        W = self._get_affinity_matrix(X,y)             # W: similarity matrix
        W_col_sum = np.sum(W, axis=1)
        n = X.shape[0]    # number of samples

        Dmhalf = np.zeros(n*n).reshape((n,n))
        for i in range(n):
            Dmhalf[i][i] = 1.0/np.sqrt(W_col_sum[i])  # D^-1/2

        DWD = np.dot(np.dot(Dmhalf,W),Dmhalf)                       # D^{-1/2}WD^{-1/2}

        m_dim = map_d
        Lambda,U = scipy.linalg.eigh(DWD,eigvals=(n-(m_dim+1),n-1))  # Spectral Decomposition

        Lambda = Lambda[::-1]   # Choose from the largest eigenvalues except for the largest
        U = U[:,::-1]
        sq_U = np.dot(U.T,U)

        Z = np.dot(Dmhalf,U)                                   # mapped points

        Z = Z[:,1:m_dim+1]  # This is for 2D display. Increase 3 to any number for dimension reduction
        
        self.Z = Z
        self.embedding = Z
        
        return self


    def fit_transform(self, X, y=None, map_d=2):
        self.fit(X,y,map_d)
        return self.embedding

    def return_C(self):
        C = self.C
        return C
    
def sinc_kernel(X,Y,alpha=1.0):             # sin(\alpha \|X-Y\|)/(pi \alpha \|X-Y\|)
    nx = X.shape[0]
    mx = X.shape[1]
    ny = Y.shape[0]
    my = Y.shape[1]
    print('(Sigmoid-Kernel ) nx ='+str(nx)+' mx='+str(mx)+' ny='+str(ny)+' my='+str(my)+' alpha='+str(alpha))
    if mx != my:
        #print('nx ='+str(nx)+' mx='+str(mx)+' ny='+str(ny)+' my='+str(my))
        exit
    A = np.zeros((nx,ny))
    for i in range(nx):
        for j in range(ny):
            nxy = np.linalg.norm(X[i,:]-Y[j,:])

            if nxy == 0.0:
                ans = 1.0/math.pi
            else:
                ans = math.sin(alpha*nxy)/(math.pi*alpha*nxy)
            A[i,j]=ans
    return A

def SLE(X,Y,la=0.3,map_d=2, n_neighbors=None): # defaul value of lambda is 0.3 to 0.4; no kernel is necessary for SLE 

    n_classes = len(np.unique(Y))
    n_samples = X.shape[0]
    #print("n_classes:",n_classes, "n_samples:",n_samples)
    
    if (n_neighbors == None):
        n_neighbors = int(1.5*n_samples/n_classes)   # 1.5 times average sample number / class

    manifolder = KSLE_ML(n_neighbors=n_neighbors, la=la, kernel="rbf")

    # Supervised Laplacian Eigenmap -> Find the mapped points
    X_transformed = manifolder.fit_transform(X,Y, map_d) 
    
    return X_transformed, manifolder