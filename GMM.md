# 高斯混合模型
GMM（高斯混合模型），高斯混合模型是另一个重要的聚类模型，相比于之前的KMeans聚类方法，使用距离作为分类的标准，在类为圆形的时候效果比较好，但是当类别为椭圆的时候，以距离为一句的聚类方法可能效果不好，所以我们需要一种更加好的聚类方法，不是基于距离的分类方法。

## 实验过程

一、 数学准备

在统计学中，正态分布是一种十分常见并且有效的分布，其他各种分布（t分布等），在中心极限定理作用下，都会趋近于正态分布。由于正态分布是一种在自然界十分常见的分布规律，所以在做预测模型的时候十分常用，解释性，广泛性极好。

正态分布数学公式：

$$
\mathcal{N}(x|\mu,\sigma) = \frac{1}{\sqrt{2\pi}\sigma} \exp(-\frac{(x - \mu)^2}{2\sigma^2})
$$

正态分布呈现中间多，两边少的分布。自然界中可以轻易地找到正态分布的应用，每个人的身高体重，成绩分布等都近似符合了正态分布。

上述公式是在一维的情况下，拓展为多维情况为：

$$
\mathcal{N}(x|\mu,\Sigma) = \frac{1}{\sqrt[d/2]{2\pi}\sqrt{|\Sigma}|} \exp(-\frac{1}{2}(x - \mu)^T\Sigma^{-1}(x-\mu))
$$


二、 算法过程

在GMM算法中，我们认为样本都来自于不同的正态分布，因此称之为高斯混合。

$$
p(x) = \sum^{K}_{k=1}\pi_k\mathcal{N}(x|\mu_k,\Sigma_k) 
$$

$$
其中\pi_k是簇k的权重，满足\sum\pi_k = 1
$$

构建计算高斯概率密度函数
```python
    def _gaussian_pdf(self, x, mu, sigma):
        size = x.shape[0]
        det = np.linalg.det(sigma)
        inv = np.linalg.inv(sigma)
        norm_const = 1.0 / (np.power((2 * np.pi), size / 2) * np.sqrt(det))
        x_mu = x - mu
        result = np.exp(-0.5 * np.dot(x_mu.T, np.dot(inv, x_mu)))
        return norm_const * result
```
根据计算公式完成密度函数计算

拟合模型（EM算法）
```python
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

            if np.abs(log_likelihood - log_likelihood_old) < self.tol:
                break
            log_likelihood_old = log_likelihood

        self.gamma = gamma
```

EM算法迭代：

$$
\text{计算后验概率矩阵，}\gamma_{ik} = \frac{\pi_k \cdot \mathcal{N}(x_i|\mu_i,\Sigma_k)}{\sum^K_{j=1}\pi_j \cdot \mathcal{N}(x_i|\mu_k,\Sigma_k)}
$$

$$
\text{更新}\mu_k,\Sigma_k,\pi_k
\mu_k = \frac{\Sigma_i\gamma_{ik}x_i}{\Sigma_i\gamma_{ik}}
\Sigma_k = \frac{\Sigma_i\gamma_{ik}(x_i-\mu_k)(x_i-\mu_k)^T}{\Sigma_i\gamma_{ik}}
$$

$$
\pi_k = \frac{\Sigma_i\gamma_{ik}}{N}
$$

$$
\text{计算似然对数} 
\log{L} = \sum_i \log(\sum_k\pi_k \cdot \mathcal{N}(x_i|\mu_i.\Sigma_k))
$$

$$
\text{终止条件}|logL_{new} - logL_{old}| < \epsilon
$$

预测
```python
def predict(self, X):
    return np.argmax(self.gamma, axis=1)

```

绘制类别
```python
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
```

## 完整代码
```python
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
```

## 总结
* 高斯混合模型利用了正态分布的想法，在解决椭圆分类的问题中，效果比KMeans更好。
* 能够处理噪声，当有离群的点的时候，KMeans的效果明显差，会严重地受到噪声的影响，但是高斯混合模型中，可以比较好的处理。
* 可能的问题：算法的前提是样本是正态分布的混合，当样本中数据严重不符合正态分布时，分类的准确性不好。