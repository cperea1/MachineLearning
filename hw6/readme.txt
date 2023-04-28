# Readme.txt

## Running the Code

To run the code, you will need to have the following libraries installed:

- `numpy`
- `pandas`
- `matplotlib`
- `sklearn`
- `scipy`

The imports that are needed for this are:

```python
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
```

Once the packages are installed, you can simply copy the code into your preferred Python environment (e.g., Jupyter Notebook, Spyder) and run it. The code is structured as follows:

```python
python hw6.py
```

I ran the code using an environment in VSCODE, and it also works via Colab. The link can be found [here](https://colab.research.google.com/drive/1JUVaGRkCUaIYpLbp3lK_j5axYbjef7OI?usp=sharing).

## Results

In summary, the analysis indicates that clustering high-dimensional datasets such as MNIST can be a difficult task, and that more sophisticated clustering algorithms may be necessary to properly group the data points.

The code first loads the Iris dataset, which is included in scikit-learn, and then applies the elbow method to determine the optimal number of clusters for the K-means algorithm. The code then applies the K-means algorithm and hierarchical clustering to the dataset with the chosen number of clusters. The agglomerative clustering algorithm is also applied to the dataset. Finally, the code evaluates the performance of the clustering algorithms using metrics such as accuracy, precision, recall, and F1 score.

The code produces various plots to visualize the results of the clustering algorithms. These plots include the SSE and silhouette scores for each value of K for the Iris dataset, the clustering results for the Iris dataset using the K-means algorithm, the dendrogram and the agglomerative clustering of the Iris dataset side by side, and the performance metrics of the clustering results for the Iris dataset.

Note that the code includes comments to explain each step of the process. Also, the runtime of each algorithm is displayed to give an idea of how long it takes to run each analysis. 

## Conclusion

This report demonstrates the importance of clustering algorithms in data analysis, and the strengths and weaknesses of different clustering methods. By applying these methods to the Iris and MNIST datasets, we gain insights into the structure of the data and the performance of the clustering algorithms.