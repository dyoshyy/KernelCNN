import numpy as np
from scipy.optimize import minimize

class GaussianKernel:
    def __init__(self, length_scale: float = 1.0):
        self.length_scale = length_scale
    
    def compute(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Compute the Gaussian kernel matrix between two sets of input data.

        Args:
            X1 (np.ndarray): First set of input data with shape (n1, d), where n1 is the number of samples and d is the dimensionality.
            X2 (np.ndarray): Second set of input data with shape (n2, d).

        Returns:
            np.ndarray: Gaussian kernel matrix with shape (n1, n2).
        """
        dist_sq = np.sum((X1[:, np.newaxis] - X2) ** 2, axis=-1)
        K = np.exp(-0.5 * dist_sq / self.length_scale**2)
        return K

class KIM:
    def __init__(self, kernel: GaussianKernel):
        self.kernel = kernel
        self.X_train = None
        self.y_train = None
        self.L = None
        self.x_mean = None
        self.x_std = None
    
    def negative_log_likelihood(self, params):
        self.kernel.length_scale = params[0]
        
        K = self.kernel.compute(self.X_train_normalized, self.X_train_normalized)
        L = np.linalg.cholesky(K + 1e-6 * np.eye(K.shape[0]))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train))
        
        nll = 0.5 * np.dot(self.y_train.T, alpha) + np.sum(np.log(np.diag(L))) + 0.5 * len(self.X_train_normalized) * np.log(2 * np.pi)
        return nll[0, 0]  # スカラー値を返すように変更
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.x_mean = np.mean(X_train, axis=0)
        self.x_std = np.std(X_train, axis=0)
        self.X_train_normalized = (X_train - self.x_mean) / self.x_std
        
        self.X_train = self.X_train_normalized
        self.y_train = y_train
        
        initial_params = np.array([self.kernel.length_scale])
        result = minimize(self.negative_log_likelihood, initial_params, method='L-BFGS-B')
        
        #print('optimized length:', result.x[0])
        self.kernel.length_scale = result.x[0]
        K = self.kernel.compute(self.X_train_normalized, self.X_train_normalized)
        self.L = np.linalg.cholesky(K + 1e-6 * np.eye(K.shape[0]))
        
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        # テストデータの正規化
        X_test_normalized = (X_test - self.x_mean) / self.x_std
        
        K_star = self.kernel.compute(X_test_normalized, self.X_train)
        v = np.linalg.solve(self.L, K_star.T)
        mean = np.dot(v.T, np.linalg.solve(self.L, self.y_train))
        return mean, None
    
