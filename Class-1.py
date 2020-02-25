from sklearn.datasets import make_blobs, make_classification, make_regression
import matplotlib.pyplot as plt

# X, y = make_blobs()
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.savefig('img.png')

# X, y = make_classification(n_informative=2, n_redundant=0, n_features=2)
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.savefig('img.png')

X, y = make_regression(n_informative=2, n_features=2)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.savefig('img.png')
