
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

np.random.seed(2)


def create_3d_data(num_classes=5,num_samples_per_class=50):
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


def visualize_data(data_3d, labels_3d, compressed_data, filename='compressed_data_visualization.png'):

    # 圧縮前の散布図
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(data_3d[:, 0], data_3d[:, 1],
                data_3d[:, 2], c=labels_3d, marker='o', s=50)
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_zlabel('Feature 3')
    ax1.set_title('Original Data')

    # 圧縮後の散布図
    ax2 = fig.add_subplot(122)
    ax2.scatter(
        compressed_data[:, 0], compressed_data[:, 1], c=labels_3d, marker='o', s=50)
    ax2.set_xlabel('Component 1')
    ax2.set_ylabel('Component 2')
    ax2.set_title('Embedded Data')

    plt.tight_layout()

    # 画像として保存
    plt.savefig(filename)
    plt.show()


def pca(X, num_components):
    # データの平均を計算して中心化
    mean = np.mean(X, axis=0)
    centered_data = X - mean

    # 共分散行列を計算
    cov_matrix = np.cov(centered_data, rowvar=False)

    # 共分散行列の固有値と固有ベクトルを計算
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 固有値の大きい順に固有ベクトルを並び替え
    sorted_indices = np.argsort(eigenvalues)[::-1]
    # sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # 上位 num_components 個の固有ベクトルを取得
    selected_eigenvectors = sorted_eigenvectors[:, :num_components]

    # 新しい特徴空間にデータを射影
    projected_data = np.dot(centered_data, selected_eigenvectors)

    return projected_data


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
        dist_list = [self.dist(x_i, x_j) for x_j in X]
        sorted_list = sorted(dist_list)  # 昇順
        threshold = sorted_list[self.k - 1]
        dist_list = np.array(dist_list)
        knn_list = dist_list <= threshold
        assert sum(knn_list) == self.k, knn_list
        return knn_list

    def dist(self, x_i, x_j):
        return np.dot(x_i - x_j, x_i - x_j)

class GPLVM(object):
    def __init__(self, θ1, θ2, θ3):
        self.θ1 = θ1
        self.θ2 = θ2
        self.θ3 = θ3

    def fit(self, X, latent_dim, epoch, eta):
        resolution = 10
        T = epoch
        N, D = X.shape
        L = latent_dim
        Z = np.random.randn(N, L) /100

        history = {}
        history['Z'] = np.zeros((T, N, L))
        history['f'] = np.zeros((T, resolution, resolution, D))
           
        for t in range(T):
            K = self.θ1 * self.kernel(Z, Z, self.θ2) +  self.θ3 * np.eye(N)
            inv_K = np.linalg.inv(K)
            dLdK = 0.5 * (inv_K @ (X @ X.T) @ inv_K - D * inv_K)
            dKdX = -2.0*(((Z[:,None,:]-Z[None,:,:])*K[:,:,None]))/self.θ2
            dLdX = np.sum(dLdK[:, :, None] *  dKdX, axis=1)

            Z = Z + eta * dLdX
            history['Z'][t] = Z

            z_new_x = np.linspace(min(Z[:,0]),max(Z[:,0]), resolution)
            z_new_y = np.linspace(min(Z[:,1]),max(Z[:,1]), resolution)
            z_new = np.dstack(np.meshgrid(z_new_x, z_new_y)).reshape(resolution**2, L)
            k_star = self.θ1 * self.kernel(z_new, Z, self.θ2) 
            F = (k_star @ inv_K @ X).reshape(resolution, resolution, D)
            history['f'][t] = F
        return history['Z'][-1]
    
    def kernel(self,X1, X2, θ2):
        Dist = np.sum(((X1[: , None, :] - X2[None, :, :])**2), axis=2)
        K = np.exp((-0.5/θ2) * Dist) 
        return K
    
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

# 3次元データの作成
X, label = create_3d_data(num_classes=7, num_samples_per_class=50)

# PCA
reduced_data = pca(X, 2)
visualize_data(X, label, reduced_data, "PCA.png")

# LE
LE = LaplacianEigenmap(2, 60)
reduced_data = LE.transform(X)
visualize_data(X, label, reduced_data, "LE.png")

# GPLVM
GPLVM_model = GPLVM(θ1=1.0, θ2=0.03, θ3=0.05)
reduced_data = GPLVM_model.fit(X,latent_dim=2, epoch=100, eta=0.0001)
visualize_data(X, label, reduced_data, "GPLVM.png")

# LPP
LPP_model = LPP(n_components=2,h=0.01)
reduced_data = LPP_model.transform(X)
visualize_data(X, label, reduced_data, "LPP.png")

