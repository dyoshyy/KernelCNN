import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from scipy.linalg import cholesky, cho_solve
plt.style.use('seaborn-pastel')

class KIM:
    def __init__(self):
        self.x_train = None
        self.train_length = None
        self.thetas = None
        self.alpha = None
        
    def objective(x):
        return 2 * np.sin(x) + 3 * np.cos(2 * x) + 5 * np.sin(2 / 3 * x)


    def rbf(self, x, x_prime, theta_1, theta_2):
        """RBF Kernel

        Args:
            x (float): data
            x_prime (float): data
            theta_1 (float): hyper parameter
            theta_2 (float): hyper parameter
        """

        return theta_1 * np.exp(-1 * np.linalg.norm(x - x_prime)**2 / theta_2)

    # Radiant Basis Kernel + Error
    def kernel(self, x, x_prime, theta_1, theta_2, theta_3, noise, eval_grad=False):
        # delta function
        if noise:
            delta = theta_3
        else:
            delta = 0

        if eval_grad:
            dk_dTheta_1 = self.kernel(x, x_prime, theta_1, theta_2, theta_3, noise) - delta
            dk_dTheta_2 = (self.kernel(x, x_prime, theta_1, theta_2, theta_3, noise) - delta) * (np.linalg.norm(x - x_prime)**2) / theta_2
            dk_dTheta_3 = delta
            return self.rbf(x, x_prime, theta_1=theta_1, theta_2=theta_2) + delta, np.array([dk_dTheta_1, dk_dTheta_2, dk_dTheta_3])

        return self.rbf(x, x_prime, theta_1=theta_1, theta_2=theta_2) + delta


    def optimize(self, x_train, y_train, bounds, initial_params=np.ones(3)):
        bounds = np.atleast_2d(bounds)

        def log_marginal_likelihood(params):
            train_length = len(x_train)
            K = np.zeros((train_length, train_length))
            for x_idx in range(train_length):
                for x_prime_idx in range(train_length):
                    K[x_idx, x_prime_idx] = self.kernel(x_train[x_idx], x_train[x_prime_idx], params[0], params[1], params[2], x_idx == x_prime_idx)

            L_ = cholesky(K, lower=True)
            alpha_ = cho_solve((L_, True), y_train)
            return - 0.5 * np.dot(y_train.T, alpha_) - np.sum(np.log(np.diag(L_))) - 0.5 * train_length * np.log(2 * np.pi)

        def log_likelihood_gradient(params):
            train_length = len(x_train)
            K = np.zeros((train_length, train_length))
            dK_dTheta = np.zeros((3, train_length, train_length))
            for x_idx in range(train_length):
                for x_prime_idx in range(train_length):
                    k, grad = self.kernel(x_train[x_idx], x_train[x_prime_idx], params[0],
                                    params[1], params[2], x_idx == x_prime_idx, eval_grad=True)
                    K[x_idx, x_prime_idx] = k
                    dK_dTheta[0, x_idx, x_prime_idx] = grad[0]
                    dK_dTheta[1, x_idx, x_prime_idx] = grad[1]
                    dK_dTheta[2, x_idx, x_prime_idx] = grad[2]

            L_ = cholesky(K, lower=True)
            alpha_ = cho_solve((L_, True), y_train)
            K_inv = cho_solve((L_, True), np.eye(K.shape[0]))
            inner_term = np.einsum("i,j->ij", alpha_, alpha_) - K_inv
            inner_term = np.einsum("ij,kjl->kil", inner_term, dK_dTheta)

            return 0.5 * np.einsum("ijj", inner_term)

        def obj_func(params):
            lml = log_marginal_likelihood(params)
            grad = log_likelihood_gradient(params)
            return -lml, -grad

        opt_res = scipy.optimize.minimize(
            obj_func,
            initial_params,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
        )

        theta_opt, func_min = opt_res.x, opt_res.fun
        return theta_opt, func_min

    
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.train_length = len(x_train)
        '''
        
        # カーネルパラメータの最適化
        self.thetas, _ = self.optimize(
            x_train, 
            y_train, 
            bounds=np.array([[1e-2, 1e2], [1e-2, 1e2], [1e-2, 1e2]]), 
            initial_params=np.array([0.5, 0.5, 0.5]))
        '''
        print("thetas: ",self.thetas)
        self.thetas = np.ones(3)

        K = np.zeros((self.train_length, self.train_length))
        for x_idx in range(self.train_length):
            for x_prime_idx in range(self.train_length):
                K[x_idx, x_prime_idx] = self.kernel(x_train[x_idx], x_train[x_prime_idx], self.thetas[0], self.thetas[1], self.thetas[2], x_idx == x_prime_idx)

        L_ = cholesky(K, lower=True)
        self.alpha_ = cho_solve((L_, True), y_train)
        
    def predict(self, x_test):
        
        test_length = len(x_test)
        mu = []
    
        for x_test_idx in range(test_length):
            k = np.zeros((self.train_length,))
            for x_idx in range(self.train_length):
                k[x_idx] = self.kernel(
                    self.x_train[x_idx],
                    x_test[x_test_idx], self.thetas[0], self.thetas[1], self.thetas[2], x_idx == x_test_idx)
            mu.append(np.dot(k, self.alpha_))
            
            #s = self.kernel(
            #    x_test[x_test_idx],
            #    x_test[x_test_idx], thetas[0], thetas[1], thetas[2], x_test_idx == x_test_idx)
            #v_ = cho_solve((L_, True), k.T)
            
            #var.append(s - np.dot(k, v_))
        return np.array(mu), None

if __name__ == '__main__':
    
    model = KIM()
    model.fit()

