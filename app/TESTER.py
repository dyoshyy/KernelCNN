import GaussianProcess as GP
import numpy as np
import matplotlib.pyplot as plt

# データセットの作成
np.random.seed(0)
n_samples = 50
X1 = np.linspace(-5, 5, n_samples)
X2 = np.linspace(-5, 5, n_samples)
X1, X2 = np.meshgrid(X1, X2)
X_train = np.column_stack((X1.ravel(), X2.ravel()))
y_train = np.sin(np.sqrt(X1**2 + X2**2)).ravel() + np.random.normal(0, 0.1, size=n_samples**2)

# ガウス過程モデルの学習と予測
gp = GP.GaussianProcess(length_scale=1.0, noise_var=0.1)
gp.fit(X_train, y_train)

n_pred = 50
X1_pred = np.linspace(-7, 7, n_pred)
X2_pred = np.linspace(-7, 7, n_pred)
X1_pred, X2_pred = np.meshgrid(X1_pred, X2_pred)
X_pred = np.column_stack((X1_pred.ravel(), X2_pred.ravel()))
y_pred = gp.predict(X_pred).reshape(n_pred, n_pred)

# プロット
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train[:, 0], X_train[:, 1], y_train, color='blue', label='Training Data')
ax.plot_surface(X1_pred, X2_pred, y_pred, cmap='jet', alpha=0.5)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y')
ax.set_title('Gaussian Process Regression')
plt.legend()
plt.savefig("GP_test.png")
