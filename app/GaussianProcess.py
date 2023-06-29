import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_solve, cholesky

class GaussianProcess:
    def __init__(self, length_scale=1.0, noise_var=0.1):
        self.length_scale = length_scale
        self.noise_var = noise_var
        self.x_train = None
        self.y_train = None
        self.alpha = None

    def kernel(self, x1, x2):
        """
        ガウシアンカーネル関数の計算
        """
        diff = x1 - x2
        sq_dist = np.dot(diff.T, diff)
        return np.exp(-0.5 * sq_dist / self.length_scale**2)

    def _negative_log_likelihood(self, params):
        """
        負の対数尤度の計算
        """
        self.length_scale, self.noise_var = np.exp(params)
        n_train = len(self.x_train)

        K = np.zeros((n_train, n_train))
        for i in range(n_train):
            for j in range(n_train):
                K[i, j] = self.kernel(self.x_train[i], self.x_train[j])

        K += self.noise_var * np.eye(n_train)

        self.L = np.linalg.cholesky(K)

        alpha = cho_solve((self.L, True), self.y_train)

        log_likelihood = -0.5 * np.dot(self.y_train.T, alpha) \
                         - np.sum(np.log(np.diag(self.L))) \
                         - 0.5 * n_train * np.log(2 * np.pi)

        return -log_likelihood[0, 0]

    def fit(self, x_train, y_train):
        """
        ガウス過程モデルの学習
        """
        self.x_train = x_train
        self.y_train = y_train

        # カーネルパラメータの最適化
        #initial_params = np.log([self.length_scale, self.noise_var])
        #result = minimize(self._negative_log_likelihood, initial_params, method='L-BFGS-B')

        #self.length_scale, self.noise_var = np.exp(result.x)

        n_train = len(x_train)
        K = np.zeros((n_train, n_train))
        for i in range(n_train):
            for j in range(n_train):
                K[i, j] = self.kernel(x_train[i], x_train[j])

        K += self.noise_var * np.eye(n_train)

        L = cholesky(K, lower=True)
        self.alpha = cho_solve((L, True), y_train)

    @profile
    def predict(self, x_pred):
        """
        予測
        """
        n_train = len(self.x_train)
        n_pred = len(x_pred)
                
        K_star = np.zeros((n_pred, n_train))
        for i in range(n_pred):
            for j in range(n_train):
                K_star[i, j] = self.kernel(x_pred[i], self.x_train[j])

        mean = np.dot(K_star, self.alpha)

        return mean
