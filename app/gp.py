import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cholesky, cho_solve

# RBF Kernel Function
def rbf_kernel(x1, x2, sigma, l):
    sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    return sigma**2 * np.exp(-0.5 / l**2 * sqdist)

# Gaussian Process Regression Class
class GaussianProcess:
    def __init__(self, kernel_func, sigma_noise=1e-3):
        self.kernel_func = kernel_func
        self.sigma_noise = sigma_noise
        self.is_fitted = False

    def compute_kernel_matrix(self, X1, X2, sigma, l):
        return self.kernel_func(X1, X2, sigma, l)

    def negative_log_marginal_likelihood(self, params, y):
        sigma, l = params
        K = self.compute_kernel_matrix(self.X_train, self.X_train, sigma, l) + self.sigma_noise**2 * np.eye(len(self.X_train))
        L = cholesky(K, lower=True)
        alpha = cho_solve((L, True), y)
        nll = 0.5 * np.dot(y.T, alpha) + np.sum(np.log(np.diagonal(L))) + len(self.X_train) * 0.5 * np.log(2 * np.pi)
        return nll

    def fit(self, X_train, y_train, sigma=1, l=1, optimize_params=False):
        self.X_train = X_train
        self.y_train = y_train
        self.models = []

        # If y_train is 1D, reshape it to 2D for consistency
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)

        for d in range(y_train.shape[1]):
            if optimize_params:
                optimized_params = minimize(self.negative_log_marginal_likelihood, [sigma, l], args=(y_train[:, d]), bounds=[(0.1, 10), (0.1, 10)], method='L-BFGS-B').x
                sigma, l = optimized_params

            K = self.compute_kernel_matrix(X_train, X_train, sigma, l) + self.sigma_noise**2 * np.eye(len(X_train))
            L = cholesky(K, lower=True)
            alpha = cho_solve((L, True), y_train[:, d])
            self.models.append((sigma, l, L, alpha))

        self.is_fitted = True

    def predict(self, X_test):
        if not self.is_fitted:
            raise ValueError("The model is not fitted yet. Please fit the model with training data before predicting.")
        
        y_pred_means = []
        y_pred_vars = []
        
        for model in self.models:
            sigma, l, L, alpha = model
            K_cross = self.compute_kernel_matrix(self.X_train, X_test, sigma, l)
            y_pred_mean = np.dot(K_cross.T, alpha)
            v = cho_solve((L, True), K_cross)
            K_test = self.compute_kernel_matrix(X_test, X_test, sigma, l)
            y_pred_var = K_test - np.dot(K_cross.T, v)
            
            y_pred_means.append(y_pred_mean)
            y_pred_vars.append(np.diag(y_pred_var))
        
        return np.array(y_pred_means).T, np.array(y_pred_vars).T

