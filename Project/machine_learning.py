# Imports for project
import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet, RANSACRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Imports from our other files
from preprocessing import *

# Set random seed
np.random.seed(42)

## MACHINE LEARNING ##

# Create a linear regression model
class linreg():
  
    # Constructor
    def __init__(self):
        self.model = LinearRegression()

    # Fits model to given data. Return fitting time
    def fitter(self, x_train, y_train): 
        tic = time.perf_counter()
        self.model.fit(x_train, y_train)
        toc = time.perf_counter()
        return (toc - tic)

    # Returns prediction on given data
    def predicter(self, x):
        return self.model.predict(x)
    
# Create a support vector regressor model
class sv_reg():
  
    # Constructor
    def __init__(self):
        self.model = SVR(C=1.0,epsilon=.2)

    # Fits model to given data. Return fitting time
    def fitter(self, x_train, y_train): 
        tic = time.perf_counter()
        self.model.fit(x_train, y_train)
        toc = time.perf_counter()
        return (toc - tic)

    # Returns prediction on given data
    def predicter(self, x):
        return self.model.predict(x)

# Create an elastic net model
class en():
  
    # Constructor
    def __init__(self):
        self.model = ElasticNet(alpha=1.0,l1_ratio=.5,random_state=42)

    # Fits model to given data. Return fitting time
    def fitter(self, x_train, y_train): 
        tic = time.perf_counter()
        self.model.fit(x_train, y_train)
        toc = time.perf_counter()
        return (toc - tic)

    # Returns prediction on given data
    def predicter(self, x):
        return self.model.predict(x)

# Create a random forest model
class randforest():
  
    # Constructor
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=10000,criterion='squared_error',random_state=42)

    # Fits model to given data. Return fitting time
    def fitter(self, x_train, y_train): 
        tic = time.perf_counter()
        self.model.fit(x_train, y_train)
        toc = time.perf_counter()
        return (toc - tic)

    # Returns prediction on given data
    def predicter(self, x):
        return self.model.predict(x)
    
    def impact(self):
        return self.model.feature_importances_
    
    def std(self):
        return np.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)

# Create a RANSAC regressor model
class sac():
  
    # Constructor
    def __init__(self):
        self.model = RANSACRegressor(LinearRegression(),max_trials=1000,min_samples=50,loss='absolute_error',
                            residual_threshold=5.0, random_state=42)

    # Fits model to given data. Return fitting time
    def fitter(self, x_train, y_train): 
        tic = time.perf_counter()
        self.model.fit(x_train, y_train)
        toc = time.perf_counter()
        return (toc - tic)

    # Returns prediction on given data
    def predicter(self, x):
        return self.model.predict(x)

def main():
    
    # Testing Code
    
    # Load Data
    usAB = load_data(1)
    nyAB = load_data(2)
    chAB = load_data(3)
    
    # Transfor the data for modeling
    usAB = get_clean_data(usAB)
    nyAB = get_clean_data(nyAB)
    chAB = get_clean_data(chAB)
    
    # Split the data for training and testing
    x_train,x_test,y_train,y_test = split_data(usAB)
    ny_x_train,ny_x_test,ny_y_train,ny_y_test = split_data(nyAB)
    ch_x_train,ch_x_test,ch_y_train,ch_y_test = split_data(chAB)
    
    # get the values for training and testing
    x_train,x_test,y_train,y_test = value_data(usAB)
    ny_x_train,ny_x_test,ny_y_train,ny_y_test = value_data(nyAB)
    ch_x_train,ch_x_test,ch_y_train,ch_y_test = value_data(chAB)
    
    
    # Scale the data for standardization of the features
    x_train_std, x_test_std = std_data(x_train,x_test)
    ny_x_train_std, ny_x_test_std = std_data(ny_x_train,ny_x_test)
    ch_x_train_std, ch_x_test_std = std_data(ch_x_train,ch_x_test)
    
    # Run the PCA for dimension reduction
    x_train_pca,x_test_pca = pca(x_train_std,x_test_std)
    ny_x_train_pca,ny_x_test_pca = pca(ny_x_train_std,ny_x_test_std)
    ch_x_train_pca,ch_x_test_pca = pca(ch_x_train_std,ch_x_test_std)
    
    # Run the KPCA for dimension Reduction
    # Elastic Net Linear Regression with KPCA
    x_train_kpca,x_test_kpca = kpca(x_train_std,x_test_std)
    # US data set throws an error because it is too large
    
    ny_x_train_kpca,ny_x_test_kpca = kpca(ny_x_train_std,ny_x_test_std)
    ch_x_train_kpca,ch_x_test_kpca = kpca(ch_x_train_std,ch_x_test_std)

 #---------------------------------------------------#   
    # Elastic Net Linear Regression
    us_en = en(x_train_std,y_train,x_test_std,y_test)
    ny_en = en(ny_x_train_std,ny_y_train,ny_x_test_std,ny_y_test)
    ch_en = en(ch_x_train_std,ch_y_train,ch_x_test_std,ch_y_test)
    
    print(us_en,ny_en,ch_en)
    
    # Elastic Net Linear Regression with PCA
    us_en_pca = en(x_train_pca,y_train,x_test_pca,y_test)
    ny_en_pca = en(ny_x_train_pca,ny_y_train,ny_x_test_pca,ny_y_test)
    ch_en_pca = en(ch_x_train_pca,ch_y_train,ch_x_test_pca,ch_y_test)
        
    print(us_en_pca,ny_en_pca,ch_en_pca)
    
    # Elastic Net Linear Regression with KPCA
    us_en_pca = en(x_train_pca,y_train,x_test_pca,y_test)
    # Cant run the KPCA for US yet
    
    ny_en_kpca = en(ny_x_train_kpca,ny_y_train,ny_x_test_kpca,ny_y_test)
    ch_en_kpca = en(ch_x_train_kpca,ch_y_train,ch_x_test_kpca,ch_y_test)
    
    print(ny_en_kpca,ch_en_kpca)
    
#---------------------------------------------------#
    # SVR Non Linear
    us_svr = sv_reg(x_train_std,y_train,x_test_std,y_test)
    ny_svr = sv_reg(ny_x_train_std,ny_y_train,ny_x_test_std,ny_y_test)
    ch_svr = sv_reg(ch_x_train_std,ch_y_train,ch_x_test_std,ch_y_test)
    
    print(us_svr,ny_svr,ch_svr)
    
    # SVR Non Linear with PCA
    us_svr_pca = sv_reg(x_train_pca,y_train,x_test_pca,y_test)
    ny_svr_pca = sv_reg(ny_x_train_pca,ny_y_train,ny_x_test_pca,ny_y_test)
    ch_svr_pca = sv_reg(ch_x_train_pca,ch_y_train,ch_x_test_pca,ch_y_test)
    
    print(us_svr_pca,ny_svr_pca,ch_svr_pca)
    
    # SVR Non Linear with KPCA
    us_svr_kpca = sv_reg(x_train_kpca,y_train,x_test_kpca,y_test)
    ny_svr_kpca = sv_reg(ny_x_train_kpca,ny_y_train,ny_x_test_kpca,ny_y_test)
    ch_svr_kpca = sv_reg(ch_x_train_kpca,ch_y_train,ch_x_test_kpca,ch_y_test)
    
    print(us_svr_kpca,ny_svr_kpca,ch_svr_kpca)

#---------------------------------------------------#
    # Random Forest Regressor Non Linear
    us_forest = randforest(x_train_std,y_train,x_test_std,y_test)
    ny_forest = randforest(ny_x_train_std,ny_y_train,ny_x_test_std,ny_y_test)
    ch_forest = randforest(ch_x_train_std,ch_y_train,ch_x_test_std,ch_y_test)
    
    print(us_forest,ny_forest,ch_forest)

    # Random Forest Regressor Non Linear with PCA
    us_forest_pca = randforest(x_train_pca,y_train,x_test_pca,y_test)
    ny_forest_pca = randforest(ny_x_train_pca,ny_y_train,ny_x_test_pca,ny_y_test)
    ch_forest_pca = randforest(ch_x_train_pca,ch_y_train,ch_x_test_pca,ch_y_test)
    
    print(us_forest_pca,ny_forest_pca,ch_forest_pca)
    
    # Random Forest Regressor Non Linear with KPCA
    us_forest_kpca = randforest(x_train_kpca,y_train,x_test_kpca,y_test)
    ny_forest_kpca = randforest(ny_x_train_kpca,ny_y_train,ny_x_test_kpca,ny_y_test)
    ch_forest_kpca = randforest(ch_x_train_kpca,ch_y_train,ch_x_test_kpca,ch_y_test)
    
    print(us_forest_kpca,ny_forest_kpca,ch_forest_kpca)

#---------------------------------------------------#
    # RANSAC Regressor Linear
    us_ransac = sac(x_train_std,y_train,x_test_std,y_test)
    ny_ransac = sac(ny_x_train_std,ny_y_train,ny_x_test_std,ny_y_test)
    ch_ransac = sac(ch_x_train_std,ch_y_train,ch_x_test_std,ch_y_test)
    
    print(us_ransac,ny_ransac,ch_ransac)

    # RANSAC Regressor Linear with PCA
    us_ransac_pca = sac(x_train_pca,y_train,x_test_pca,y_test)
    ny_ransac_pca = sac(ny_x_train_pca,ny_y_train,ny_x_test_pca,ny_y_test)
    ch_ransac_pca = sac(ch_x_train_pca,ch_y_train,ch_x_test_pca,ch_y_test)
    
    print(us_ransac_pca,ny_ransac_pca,ch_ransac_pca)
    
    # RANSAC Regressor Linear with KPCA
    us_ransac_kpca = sac(x_train_kpca,y_train,x_test_kpca,y_test)
    ny_ransac_kpca = sac(ny_x_train_kpca,ny_y_train,ny_x_test_kpca,ny_y_test)
    ch_ransac_kpca = sac(ch_x_train_kpca,ch_y_train,ch_x_test_kpca,ch_y_test)
    
    print(us_ransac_kpca,ny_ransac_kpca,ch_ransac_kpca)
#---------------------------------------------------#


    
if __name__ == '__main__':
    main() 
