#Calicia Perea
#hw 5: Regression 
# April 7, 2023

# Each regressor needs to be tested using the California housing dataset, which can
# be loaded using fetch_california_housing from sklearn.datasets. You need to use all the
# columns and all the instances in this dataset. Such analysis should include at least Mean squared
#error (MSE), R2 score, and the fitting (or training) time. 
import numpy as np
import time
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, RANSACRegressor,Ridge
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# load the dataset 
X,y = fetch_california_housing(return_X_y= True)

#split the dataset into traing and testing 
X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size =0.2, random_state = 42
)

# create each regression function
regressor = [
    LinearRegression(),
    RANSACRegressor(),
    Ridge(),
    Lasso(),
    ElasticNet()
]

#Train and evaluate each regression function 
for function in regressor:
    start = time.perf_counter()
    function.fit(X_train, y_train)
    stop = time.perf_counter()
    fit_time = stop - start
    y_pred = function.predict(X_test)
    mse = mean_squared_error(y_test,y_pred)
    r2 = r2_score(y_test, y_pred)
    plt.scatter(y_test, y_pred, label=function.__class__.__name__ 
                + " MSE:" + str(round(mse, 2)) + " R2:" 
                + str(round(r2, 2)) + " Fit time:" 
                + str(round(fit_time, 2)))
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs Actual Values for Regression Functions")
    plt.legend()
    plt.show()
    print(function.__class__.__name__, " MSE:", mse," R2:",r2,
        " Fit time:", fit_time)