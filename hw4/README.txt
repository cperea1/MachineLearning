Calicia Perea 
HW4- Machine Learning 
Readme.txt

This code compares the performance of three dimensionality reduction techniques - PCA, LDA, and KernelPCA - by feeding the dimension-reduced data to a decision tree classifier and checking the classification results on the Iris and MNIST datasets. The accuracy of each method is printed, as well as the fitting time of each method. The results show that LDA had the highest accuracy on the Iris dataset and the second-highest on the MNIST dataset, while PCA had the lowest accuracy on both datasets. Modifying the parameters slightly improved the accuracies, but increasing the number of components did not show any improvements.
To try different parameters, you can modify the values used to initialize each dimensionality reduction technique. For example, you can change the number of components used by PCA and LDA, the kernel function and gamma value used by KernelPCA, and so on. The code can be modified to fit and evaluate the new results.
The code was tested on both VS Code and Google Colab, but an output error was encountered. The error did not affect the accuracy results of the models, but we do not know how to fix it in our environment.
The code is written in Python and requires the following libraries to be installed:
	•	scikit-learn
	•	numpy
To run the code, simply execute the script in a Python environment with the required libraries installed.
