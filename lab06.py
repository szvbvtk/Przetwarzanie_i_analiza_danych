# from sklearn import datasets
# from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import pandas as pd
import numpy.random as rnd
# from sklearn import decomposition
# import seaborn as sns; sns.set()
# from sklearn.datasets import load_digits

# iris = datasets.load_iris()
# pca = PCA(n_components=2)
# x=iris.data
# y = iris.target
# target_names = iris.target_names
# X_r = pca.fit(x).transform(x)

# plt.figure()
# kolory = ["red", "green", "blue"]
# lw = 2

# for color, i, target_name in zip(kolory, [0, 1, 2], target_names):
#     plt.scatter(
#         X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
#     )
# plt.legend(loc="best", shadow=False, scatterpoints=1)

# plt.show()


x = range(200) + np.random.randint(0,30,200)
y = range(200) + np.random.randint(0,30,200)

plt.scatter(x, y)

plt.show()
Y = np.random.randint(10,50,100).reshape(20,5)

# sortowanie wektorów i wartości

def PCA(X, num_components):
    X_średnie = X - np.mean(X, axis=0)
    cov_mat = np.cov(X_średnie, rowvar=False)
    osobne_wartosci, osobne_wektory = np.linalg.eigh(cov_mat)
    posortowany_indeks = np.argsort(osobne_wartosci)[::-1]
    posortowane_osobne_wartosci = osobne_wartosci[posortowany_indeks]
    posortowane_osobne_wektory = osobne_wektory[:, posortowany_indeks]
    eigenvector_subset = posortowane_osobne_wektory[:, 0:num_components]
    X_zredukowanee = np.dot(eigenvector_subset.transpose(), X_średnie.transpose()).transpose()
    return X_zredukowanee

ndim = 2
mu = np.random.randint([10] * ndim) # srednia
sigma = np.zeros((ndim, ndim)) - 1.8 # kowariancja
np.fill_diagonal(sigma, 3.5)
print("Mu ", mu.shape)
print("Sigma", sigma.shape)


oryginalne_dane = rnd.multivariate_normal(mu, sigma, size=(200))
print("Kształt danych:", oryginalne_dane.shape)

x = np.random.random((50,4))
y = np.random.random((4, 8))

dot = np.dot(x, y)
print(y)



zredukowane_x = PCA(dot, 1)
print("\nWypisanie wartości z pca: ", zredukowane_x, "\n")


rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
plt.scatter(X[:, 0], X[:, 1])
plt.axis('equal')

zredukowane_x = PCA(X, 2)

plt.scatter(X[:, 0], X[:, 1])
plt.scatter(zredukowane_x, zredukowane_x)
plt.axis('equal')

plt.show()

#2
# iris = datasets.load_iris()
# x=iris.data

# rng = np.random.RandomState(1)
# X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
# plt.scatter(x[:, 0], x[:, 1])
# plt.axis('equal')

# zredukowane_x = PCA(X, 2)

# plt.scatter(x[:, 0], x[:, 1],marker='*')
# plt.scatter(zredukowane_x, zredukowane_x,marker='+')
# plt.axis('equal')

# plt.show()

#3
# digits = load_digits()
# x=digits

# zredukowane_x = PCA(X, 2)

# plt.scatter(zredukowane_x, zredukowane_x,marker='+',color='black')
# plt.axis('equal')

# plt.show()