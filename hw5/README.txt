Calicia Perea 
HW5- Machine Learning 
Readme.txt

This code performs linear regression analysis on the California housing dataset using scikit-learn library. The dataset can be loaded using fetch_california_housing from sklearn.datasets.

The following regression functions are used:

1. LinearRegression
2. RANSACRegressor
3. Ridge
4. Lasso
5. ElasticNet

The analysis includes at least Mean squared error (MSE), R2 score, and the fitting (or training) time.

The code outputs the mean squared error (MSE), R2 score, scatterplot, and fit time for each regression function on the dataset. The scatter plots show that LinearRegression and Ridge models have a better fit than the other models. The code output shows the mean squared error (MSE), R2 score, and fit time for each of the five regression functions used to model the California housing dataset. The LinearRegression and Ridge models have the lowest MSE and highest R2 score, indicating their better performance in predicting the target variable. RANSACRegressor has the highest MSE and negative R2 score, indicating poor performance in predicting the target variable. The Lasso and ElasticNet models have an intermediate performance, with Lasso having a higher MSE and lower R2 score compared to ElasticNet. The fit time for each model is also displayed, indicating how long it took to train the model.


The code is written in Python and requires the following libraries to be installed:
	•	scikit-learn
	•	numpy
	•	matplotlib

To run the code, simply execute the script in a Python environment with the required libraries installed.
