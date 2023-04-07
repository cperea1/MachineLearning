#Calicia Perea
#March 26 2023
#HW4 Dimensionality reduction techniques

from sklearn.datasets import load_iris, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load Iris dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# Load MNIST dataset
mnist = fetch_openml("mnist_784")
X_mnist = mnist.data.astype('float64')
y_mnist = mnist.target.astype('int64')

# Split MNIST dataset to create subset for KernelPCA
X_mnist_sub, _, y_mnist_sub, _ = train_test_split(
    X_mnist, y_mnist, stratify=y_mnist, 
    test_size=0.9, random_state=42)

# PCA
pca = PCA(n_components=2)
X_iris_pca = pca.fit_transform(X_iris)
X_mnist_pca = pca.fit_transform(X_mnist)

# LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_iris_lda = lda.fit_transform(X_iris, y_iris)
X_mnist_lda = lda.fit_transform(X_mnist, y_mnist)

# Kernel PCA
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
X_mnist_kpca = kpca.fit_transform(X_mnist_sub)

# Decision Tree Classifier
clf = DecisionTreeClassifier()

# Fit and evaluate on Iris dataset
clf.fit(X_iris_pca, y_iris)
print("PCA accuracy on Iris dataset:", clf.score(X_iris_pca, y_iris))
clf.fit(X_iris_lda, y_iris)
print("LDA accuracy on Iris dataset:", clf.score(X_iris_lda, y_iris))

# Fit and evaluate on MNIST dataset
clf.fit(X_mnist_pca, y_mnist)
print("PCA accuracy on MNIST dataset:", clf.score(X_mnist_pca, y_mnist))
clf.fit(X_mnist_lda, y_mnist)
print("LDA accuracy on MNIST dataset:", clf.score(X_mnist_lda, y_mnist))
clf.fit(X_mnist_kpca, y_mnist_sub)
print("Kernel PCA accuracy on MNIST dataset:", clf.score(X_mnist_kpca, y_mnist_sub))