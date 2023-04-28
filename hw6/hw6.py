import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, fetch_openml
from sklearn.metrics import *
from sklearn.metrics import silhouette_score, precision_score, recall_score, f1_score,confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import dendrogram, linkage
import time
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering


#--------------IRIS----------------------------------------------
divider = '-' * 20  # create a divider string
for i in range(2):
    print('\n' + divider + '\n')  # print new lines with the divider in between



# Load the Iris dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target



#-----Elbow method to decide a reasonable K for the K-means algorithm.--------- 
# Apply the elbow method to determine the optimal number of clusters
start_time = time.time()
sse = []
silhouette_scores = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_iris)
    sse.append(kmeans.inertia_)
    if k > 1:
        silhouette_scores.append(silhouette_score(X_iris, kmeans.labels_))
end_time = time.time()
elbow_time = end_time - start_time

divider = '-' * 20  # create a divider string
for i in range(2):
    print('\n' + divider + '\n')  # print new lines with the divider in between
print(f"Elbow method runtime: {elbow_time:.2f} seconds")

divider = '-' * 20  # create a divider string
for i in range(1):
    print('\n' + divider + '\n')  # print new lines with the divider in between

# Plot the SSE and silhouette scores for each value of K for the Iris dataset
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(range(1, 11), sse)
ax[0].set_xlabel('Number of clusters')
ax[0].set_ylabel('SSE')
ax[0].set_title('Elbow Method for Iris Dataset')
ax[1].plot(range(2, 11), silhouette_scores)
ax[1].set_xlabel('Number of clusters')
ax[1].set_ylabel('Silhouette Score')
ax[1].set_title('Silhouette Scores for Iris Dataset')
plt.show()

# Apply the K-means clustering algorithm to the Iris dataset with the chosen number of clusters
start_time = time.time()
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_iris)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
kmeans_time = time.time() - start_time



print(f"K-means runtime: {kmeans_time:.2f} seconds")

divider = '-' * 20  # create a divider string
for i in range(1):
    print('\n' + divider + '\n')  # print new lines with the divider in between



# Plot the clustering results for the Iris dataset
plt.scatter(X_iris[:, 0], X_iris[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=300, c='r')
plt.title('K-means Clustering for Iris Dataset')
plt.show()

# Apply the hierarchical clustering algorithm to the Iris dataset
start_time = time.time()

# Apply the hierarchical clustering algorithm to the Iris dataset
Z = linkage(X_iris, 'ward')

hi_time = time.time() - start_time
print(f"Hierarchical runtime: {hi_time:.2f} seconds")

# Apply the agglomerative clustering algorithm to the Iris dataset
start_time_a = time.time()
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
agg_labels = agg.fit_predict(X_iris)

agg_time = time.time() - start_time_a
print(f"Agglomerative runtime: {agg_time:.2f} seconds")


# Plot the dendrogram and the agglomerative clustering of the Iris dataset side by side
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))

# Plot dendrogram
dendrogram(Z, truncate_mode='lastp', p=20, ax=ax1)
ax1.set(title='Hierarchical Clustering for Iris Dataset')

# Plot agglomerative clustering
colors = np.array(['r', 'g', 'b'])
ax2.scatter(X_iris[:, 0], X_iris[:, 1], c=colors[agg_labels])
ax2.set(title='Agglomerative Clustering for Iris Dataset')

plt.show()



divider = '-' * 20  # create a divider string
for i in range(1):
    print('\n' + divider + '\n')  # print new lines with the divider in between


# Calculate the accuracy, precision, recall, and F1 score of the clustering results for the Iris dataset
start_time = time.time()
accuracy = np.mean(labels == y_iris)
precision = precision_score(y_iris, labels, average='macro')
recall = recall_score(y_iris, labels, average='macro')
f1 = f1_score(y_iris, labels, average='macro')

ac_time = time.time() - start_time

print(f"Performance metrics runtime: {ac_time:.2f} seconds")
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

divider = '-' * 20  # create a divider string
for i in range(1):
    print('\n' + divider + '\n')  # print new lines with the divider in between

# doing analysis based off of class labels as ground truth
start_time = time.time()
# Create DataFrame with features
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Create KMeans object
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit KMeans to data
kmeans.fit(iris_df)

# Predict labels
labels = kmeans.predict(iris_df)

# Create confusion matrix
cm = confusion_matrix(iris.target, labels)
end_time = time.time()
cli_time = end_time- start_time
print(f"Class labels as ground truth metrics runtime: {cli_time:.2f} seconds")
# Plot confusion matrix
plt.imshow(cm, cmap='binary')
plt.xticks(ticks=[0,1,2], labels=iris.target_names)
plt.yticks(ticks=[0,1,2], labels=iris.target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.colorbar()
plt.show()

divider = '-' * 20  # create a divider string
for i in range(1):
    print('\n' + divider + '\n')  # print new lines with the divider in between


# printing the description of the dataset
print("/n" + iris.DESCR)

divider = '-' * 20  # create a divider string
for i in range(3):
    print('\n' + divider + '\n')  # print new lines with the divider in between

#--------------MINST----------------------------------------------


# Load the MNIST dataset and extract a subset
mnist = fetch_openml('mnist_784', version=1)

# Split the dataset into train and test sets, and extract a subset from the train set
X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.8, stratify=mnist.target, random_state=42)

# randomly select 2000 indices
indices = np.random.choice(X_train.index,size = 2000, replace=False)

X_train_subset = X_train.loc[indices].astype(int)
y_train_array = y_train.loc[indices].astype(int).values



# Reshape the images from 2D to 1D arrays
X_train_1d = X_train_subset.values.reshape(X_train_subset.shape[0], -1)

# for further analysis
# Apply PCA to reduce the dimensionality of the MNIST dataset
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train_1d)


# Apply the elbow method to determine the optimal number of clusters
start_time = time.time()
sse = []
silhouette_scores = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train_pca)
    sse.append(kmeans.inertia_)
    if k > 1:
        silhouette_scores.append(silhouette_score(X_train_pca, kmeans.labels_))
end_time = time.time()
elbow_time = end_time - start_time

print(f"Elbow method runtime: {elbow_time:.2f} seconds")

divider = '-' * 20  # create a divider string
for i in range(1):
    print('\n' + divider + '\n')  # print new lines with the divider in between


# Plot the SSE and silhouette scores for each value of K for the MNIST dataset
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(range(1, 11), sse)
ax[0].set_xlabel('Number of clusters')
ax[0].set_ylabel('SSE')
ax[0].set_title('Elbow Method for MNIST Dataset')
ax[1].plot(range(2, 11), silhouette_scores)
ax[1].set_xlabel('Number of clusters')
ax[1].set_ylabel('Silhouette Score')
ax[1].set_title('Silhouette Scores for MNIST Dataset')
plt.show()

# Apply the K-means clustering algorithm to the MNIST dataset with the chosen number of clusters
start_time = time.time()
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(X_train_pca)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
kmeans_time = time.time() - start_time



print(f"K-means runtime: {kmeans_time:.2f} seconds")

divider = '-' * 20  # create a divider string
for i in range(1):
    print('\n' + divider + '\n')  # print new lines with the divider in between

# Plot the clustering results for the MNIST dataset
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=labels)
ax.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=500, c='r')
ax.set_title('K-means Clustering for MNIST Dataset')
plt.show()


# Apply the hierarchical clustering algorithm to the MNIST dataset
start_time = time.time()
Z = linkage(X_train_pca, 'ward')


hi_time = time.time() - start_time
print(f"Hierarchical runtime: {hi_time:.2f} seconds")

# Apply the agglomerative clustering algorithm to the MNIST dataset
start_time = time.time()
agg = AgglomerativeClustering(n_clusters=10, linkage='ward')
agg_labels = agg.fit_predict(X_train_pca)

agg_time = time.time() - start_time
print(f"Agglomerative runtime: {agg_time:.2f} seconds")

# Plot the dendrogram and the agglomerative clustering of the MNIST dataset side by side
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))

# Plot dendrogram
dendrogram(Z, truncate_mode='lastp', p=20, ax=ax1)
ax1.set(title='Hierarchical Clustering for MNIST Dataset')

# Plot agglomerative clustering
ax2.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=agg_labels)
ax2.set(title='Agglomerative Clustering for MNIST Dataset')

plt.show()


divider = '-' * 20  # create a divider string
for i in range(1):
    print('\n' + divider + '\n')  # print new lines with the divider in between

# Calculate the accuracy, precision, recall, and F1 score of the clustering results for the MNIST dataset
start_time = time.time()

from sklearn.preprocessing import LabelEncoder

# Convert string labels to numeric labels
le = LabelEncoder()
le.fit(y_train_array)
labels_numeric = le.transform(labels)

# calculate precision score, accuracy, recall, and f1
precision = precision_score(y_train_array, labels_numeric, average='macro')
accuracy = np.mean(labels_numeric == y_train_array)
recall = recall_score(y_train_array, labels_numeric, average='macro')
f1 = f1_score(y_train_array, labels_numeric, average='macro')

ac_time = time.time() - start_time
# Print the performance metrics and running time of the clustering algorithm for the MNIST dataset
print(f"Performance metrics runtime: {ac_time:.2f} seconds")
print(f"Accuracy: {accuracy:.4f},  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


divider = '-' * 20  # create a divider string
for i in range(1):
    print('\n' + divider + '\n')  # print new lines with the divider in between

# doing analysis based off of class labels as ground truth
start_time = time.time()
# Extract data and labels
X = mnist.data
y = mnist.target.astype(int)

# Create KMeans object
kmeans = KMeans(n_clusters=10, random_state=42)

# Fit KMeans to data
kmeans.fit(X)

# Predict labels
labels = kmeans.predict(X)

# Create confusion matrix
cm = confusion_matrix(y, labels)
end_time = time.time()
clm_time = end_time- start_time
print(f"Class labels as ground truth metrics runtime: {clm_time:.2f} seconds")

# Plot confusion matrix
plt.imshow(cm, cmap='binary')
plt.xticks(ticks=[0,1,2,3,4,5,6,7,8,9], labels=[str(i) for i in range(10)])
plt.yticks(ticks=[0,1,2,3,4,5,6,7,8,9], labels=[str(i) for i in range(10)])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.colorbar()
plt.show()

divider = '-' * 20  # create a divider string
for i in range(1):
    print('\n' + divider + '\n')  # print new lines with the divider in between


# printing the description of the dataset
print("/n" + mnist.DESCR)

divider = '-' * 20  # create a divider string
for i in range(3):
    print('\n' + divider + '\n')  # print new lines with the divider in between


