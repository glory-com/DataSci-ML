import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class GMM:
    def __init__(self, n_components=2, max_iters=100, tol=1e-4, random_state=None):
        self.K = n_components   
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state

        self.mu = None
        self.sigma = None
        self.pi = None
        self.gamma = None

    def _gaussian_pdf(self, x, mu, sigma):
        size = x.shape[0]
        det = np.linalg.det(sigma)
        inv = np.linalg.inv(sigma)
        norm_const = 1.0 / (np.power((2 * np.pi), size / 2) * np.sqrt(det))
        x_mu = x - mu
        result = np.exp(-0.5 * np.dot(x_mu.T, np.dot(inv, x_mu)))
        return norm_const * result

    def fit(self, X):
        np.random.seed(self.random_state)
        N, D = X.shape
        self.mu = X[np.random.choice(N, self.K, replace=False)]
        self.sigma = [np.eye(D) for _ in range(self.K)]
        self.pi = np.ones(self.K) / self.K

        log_likelihood_old = 0

        for iteration in range(self.max_iters):
            gamma = np.zeros((N, self.K))
            for k in range(self.K):
                for i in range(N):
                    gamma[i, k] = self.pi[k] * self._gaussian_pdf(X[i], self.mu[k], self.sigma[k])
            gamma /= np.sum(gamma, axis=1, keepdims=True)

            N_k = np.sum(gamma, axis=0)
            for k in range(self.K):
                self.mu[k] = np.sum(gamma[:, k, np.newaxis] * X, axis=0) / N_k[k]
                diff = X - self.mu[k]
                self.sigma[k] = (gamma[:, k, np.newaxis] * diff).T @ diff / N_k[k]
                self.pi[k] = N_k[k] / N

            log_likelihood = 0
            for i in range(N):
                temp = 0
                for k in range(self.K):
                    temp += self.pi[k] * self._gaussian_pdf(X[i], self.mu[k], self.sigma[k])
                log_likelihood += np.log(temp + 1e-10)

            print(f"迭代 {iteration+1}, 对数似然: {log_likelihood:.6f}")

            if np.abs(log_likelihood - log_likelihood_old) < self.tol:
                break
            log_likelihood_old = log_likelihood

        self.gamma = gamma

    def predict(self, X):
        return np.argmax(self.gamma, axis=1)

    def plot_ellipse(self, ax, mu, sigma, color):
        eigenvalues, eigenvectors = np.linalg.eigh(sigma)
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width, height = 2 * np.sqrt(eigenvalues)
        ell = Ellipse(xy=mu, width=width, height=height, angle=angle, color=color, alpha=0.3)
        ax.add_patch(ell)

    def plot(self, X):
        cluster_labels = self.predict(X)
        colors = ["blue", "green", "red", "orange", "black", "purple", "cyan", "magenta"]

        fig, ax = plt.subplots()
        for k in range(self.K):
            ax.scatter(X[cluster_labels == k, 0], X[cluster_labels == k, 1], c=colors[k % len(colors)], label=f"Cluster {k+1}")
            ax.scatter(self.mu[k][0], self.mu[k][1], c="red", marker="x", s=100)
            self.plot_ellipse(ax, self.mu[k], self.sigma[k], colors[k % len(colors)])

        plt.show()
        
data = pd.read_csv("data_class.csv")
X = data[["X", "Y"]].values

gmm = GMM(n_components=4, max_iters=100, tol=1e-4, random_state=42)
gmm.fit(X)
gmm.plot(X)