
# --------------------------------------
# |          |     線形    |   非線形   |
# --------------------------------------
# | 教師あり  |    LDA     |            |
# --------------------------------------
# | 教師無し  |    PCA     |     LE     |
# --------------------------------------

# 線形埋め込み：PrincipalComponentsAnalysis
# 非線形埋め込み：Laplacian Eigenmap
# 教師あり：
# 教師無し：

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import random
from tqdm import tqdm

np.set_printoptions(threshold=np.inf)
np.random.seed(2)


def create_3d_data(num_classes=5, num_samples_per_class=50):
    # データのクラス数と特徴量の次元数
    num_features = 3

    data = []
    labels = []
    for i in range(num_classes):
        # 各クラスの中心点（3次元空間上のランダムな座標）
        center = np.random.rand(num_features)

        # 各クラスのデータを生成
        class_data = center + \
            np.random.randn(num_samples_per_class, num_features) * 0.1
        data.append(class_data)
        labels.extend([i] * num_samples_per_class)

    data = np.vstack(data)
    labels = np.array(labels)

    return data, labels


def visualize_data(labels_3d, compressed_data, filename='compressed_data_visualization.png'):

    # 圧縮前の散布図
    fig = plt.figure(figsize=(6, 6))
    '''
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(data_3d[:, 0], data_3d[:, 1],
                data_3d[:, 2], c=labels_3d, marker='o', s=50)
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_zlabel('Feature 3')
    ax1.set_title('Original Data')
    '''
    # 圧縮後の散布図
    ax2 = fig.add_subplot(111)
    ax2.scatter(
        compressed_data[:, 0], compressed_data[:, 1], c=labels_3d, marker='o', s=50)
    ax2.set_xlabel('Component 1')
    ax2.set_ylabel('Component 2')
    ax2.set_title('Embedded Data')

    plt.tight_layout()

    # 画像として保存
    plt.savefig(filename)
    plt.show()


def PCA(X, num_components):
    mean = np.mean(X, axis=0)
    centered_data = X - mean

    cov_matrix = np.cov(centered_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    selected_eigenvectors = sorted_eigenvectors[:, :num_components]
    projected_data = np.dot(centered_data, selected_eigenvectors)

    return projected_data

class GPLVM(object):
    def __init__(self,Y,LatentDim,HyperParam,X=None):
        self.Y = Y
        self.hyperparam = HyperParam
        self.dataNum = self.Y.shape[0]
        self.dataDim = self.Y.shape[1]

        self.latentDim = LatentDim
        if X is not None:
            self.X = X
        else:
            self.X = 0.1*np.random.randn(self.dataNum,self.latentDim)
        self.S = Y @ Y.T
        self.history = {}

    def fit(self,epoch=100,epsilonX=0.5,epsilonSigma=0.0025,epsilonAlpha=0.00005):

        self.history['X'] = np.zeros((epoch, self.dataNum, self.latentDim))
        sigma = np.log(self.hyperparam[0])
        alpha = np.log(self.hyperparam[1])
        for i in tqdm(range(epoch)):

            # 潜在変数の更新
            K = self.kernel(self.X,self.X,self.hyperparam[0]) + self.hyperparam[1]*np.eye(self.dataNum)
            Kinv = np.linalg.inv(K)
            G = 0.5*(Kinv @ self.S @ Kinv-self.dataDim*Kinv)
            dKdX = -2.0*(((self.X[:,None,:]-self.X[None,:,:])*K[:,:,None]))/self.hyperparam[0]
            # dFdX = (G[:,:,None] * dKdX).sum(axis=1)-self.X
            dFdX = (G[:,:,None] * dKdX).sum(axis=1)

            # ハイパーパラメータの更新
            Dist = ((self.X[:, None, :] - self.X[None, :, :]) ** 2).sum(axis=2)
            dKdSigma = 0.5*Dist/self.hyperparam[0] * ( K- self.hyperparam[1]*np.eye(self.dataNum) )
            dFdSigma = np.trace(G @ dKdSigma)

            dKdAlpha = self.hyperparam[1]*np.eye(self.dataNum)
            dFdAlpha = np.trace(G @ dKdAlpha)

            self.X = self.X + epsilonX * dFdX
            self.history['X'][i] = self.X
            sigma = sigma + epsilonSigma * dFdSigma
            self.hyperparam[0] = np.exp(sigma)
            alpha = alpha + epsilonAlpha * dFdAlpha
            self.hyperparam[1] = np.exp(alpha)

            #K = self.kernel(self.X,self.X,self.hyperparam[0]) + self.hyperparam[1]*np.eye(self.dataNum)
            #Kinv = np.linalg.inv(K)

        return self.history['X'][-1]

    def kernel(self,X1, X2, length):
        Dist = (((X1[:, None, :] - X2[None, :, :]) ** 2) / length).sum(axis=2)
        K = np.exp(-0.5 * Dist)
        return K

class LaplacianEigenmap:
    def __init__(self, n_components, k):
        """
        param: n_component : embedding dim
        param: k : knn of similarity matrix
        """
        self.n_components = n_components
        self.k = k

    def transform(self, X):
        W = self.create_similarity_matrix(X)
        D = self.get_degree_matrix(W)
        D = D.astype(float)
        L = D - W
        eig_val, eig_vec = eigh(L, D)
        eig_vec = eig_vec.T
        index = np.argsort(eig_val)
        eig_vec = eig_vec[index]
        phi = eig_vec[1:self.n_components + 1]
        return phi.T

    def get_degree_matrix(self, W):
        return np.diag([sum(W[i]) for i in range(len(W))])

    def create_similarity_matrix(self, X):
        """create Similarity matrix (knn)

        :param X: data matrix (data_nX,feature_n)
        """
        W = []
        for x_i in X:
            W.append(self.k_nearest_list(X, x_i))
        W = np.array(W)
        return np.where(np.logical_or(W, W.T), 1, 0)

    def k_nearest_list(self, X, x_i):
        """
        return the ndarray containing bool value represents whether the value is in k nearest neighbor of x_i
        e.g. ndarray [True False True]
        """
        # print("X", X)
        dist_list = [self.dist(x_i, x_j) for x_j in X]
        # print("dist_list", dist_list)
        sorted_list = sorted(dist_list)  # 昇順
        #print("sorted_list", sorted_list)
        threshold = sorted_list[self.k - 1]
        dist_list = np.array(dist_list)
        knn_list = dist_list <= threshold
        
        # 距離が同じ点の集合を作成
        same_dist_indices = [i for i, dist in enumerate(dist_list) if dist == threshold]
        
        # もし同じ距離の点がself.kよりも多い場合はランダムサンプリング
        if sum(knn_list) > self.k:
            knn_list[same_dist_indices] = False
            random_indices = random.sample(same_dist_indices, self.k-sum(knn_list))
            knn_list[random_indices] = True

        assert sum(knn_list) == self.k, knn_list
        return knn_list

    def dist(self, x_i, x_j):
        return np.dot(x_i-x_j, x_i-x_j)

class LPP:
    """Locality Preserving Projection."""

    def __init__(self, n_components, h=1 / np.sqrt(2)):
        self.n_components = n_components
        self.h = h

    def transform(self, X):
        X = X - np.mean(X, axis=0)
        W = self.create_similarity_matrix(X)
        D = self.get_degree_matrix(W)
        L = D - W
        A = X.T @ L @ X
        B = X.T @ D @ X
        eig_val, eig_vec = eigh(A, B)
        index = np.argsort(eig_val)
        eig_vec = eig_vec[index]
        components = []
        for vec in eig_vec:  # normalize
            normalized_vec = vec / np.linalg.norm(vec)
            components.append(normalized_vec)
        self.components_ = np.array(components[:self.n_components])
        return X @ self.components_.T

    def gaussian_kernel(self, x_i, x_j):
        """kernel function"""
        return math.e**(-np.dot(x_i - x_j, x_i - x_j) / (2 * self.h**2))

    def get_degree_matrix(self, W):
        return np.diag([sum(W[i]) for i in range(len(W))])

    def create_similarity_matrix(self, X):
        """create Similarity matrix

        :param X: data matrix (data_nX,feature_n)
        """
        W = []
        for x_i in X:
            K_i = []
            for x_j in X:
                K_i.append(self.gaussian_kernel(x_i, x_j))
            W.append(K_i)
        return np.array(W)


if __name__ == '__main__':
    # 3次元データの作成
    X, label = create_3d_data(num_classes=7, num_samples_per_class=50)
    print("X shape:", np.shape(X))

    # PCA
    reduced_data = PCA(X, 2)
    visualize_data(X, label, reduced_data, "PCA.png")

    # LE
    LE = LaplacianEigenmap(2, 60)
    reduced_data = LE.transform(X)
    visualize_data(X, label, reduced_data, "LE.png")

    # GPLVM
    GPLVM_model = GPLVM(θ1=1.0, θ2=0.03, θ3=0.05)
    reduced_data = GPLVM_model.fit(X, latent_dim=2, epoch=100, eta=0.0001)
    visualize_data(X, label, reduced_data, "GPLVM.png")

    # LPP
    LPP_model = LPP(n_components=2, h=0.01)
    reduced_data = LPP_model.transform(X)
    visualize_data(X, label, reduced_data, "LPP.png")
