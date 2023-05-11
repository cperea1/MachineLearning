# Imports for project
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# Imports from our other files
from data_vis import *
from preprocessing import *
from machine_learning import *

# Ignore warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
      
## MAIN ##

def main():
        
# Load in the data for each dataset; provide associated number
    usAB = load_data(1)
    nyAB = load_data(2)
    chAB = load_data(3)

# Data visualization
    data_vis(usAB, nyAB, chAB)
    
# This function converts the dataset to all int/float or date data types and removes missing values 
    usAB = get_clean_data(usAB)
    nyAB = get_clean_data(nyAB)
    chAB = get_clean_data(chAB)
    
# Heat Maps
    corr_heat_map(usAB)
    corr_heat_map(nyAB)
    corr_heat_map(chAB)

# Pairplots
    pairplot(usAB)
    pairplot(nyAB)
    pairplot(chAB)     
           
# This function will split the data and return unscaled data
# just input the data set you want to use in the function and 
# make sure you have the number of variables to store the return into

    x_train,x_test,y_train,y_test = split_data(usAB)
    ny_x_train,ny_x_test,ny_y_train,ny_y_test = split_data(nyAB)
    ch_x_train,ch_x_test,ch_y_train,ch_y_test = split_data(chAB)

# standardized the data using standard scaler
# need two variables to store the returning data

    x_train_std, x_test_std = std_data(x_train,x_test)
    ny_x_train_std, ny_x_test_std = std_data(ny_x_train,ny_x_test)
    ch_x_train_std, ch_x_test_std = std_data(ch_x_train,ch_x_test)

# Run the Random Forest Regressor on each dataset


    # Calculate US data and print results
    rf_us = randforest()
    rf_us_time = rf_us.fitter(x_train_std, y_train)
    train_pred = rf_us.predicter(x_train_std)
    test_pred = rf_us.predicter(x_test_std)
    rf_us_r2 = r2_score(y_test, test_pred)
    print("\n\t-- US Data --")
    print(f"Fitting Time:  {rf_us_time:.2} seconds")
    print("Training R2:  ", r2_score(y_train, train_pred))
    print("Training MSE: ", mean_squared_error(y_train, train_pred))
    print("Testing R2:   ", r2_score(y_test, test_pred))
    print("Testing MSE:  ", mean_squared_error(y_test, test_pred))
    residual_plot(y_train, y_test, train_pred, test_pred, "Random Forest (US Data)")

    # Extract the feature importance and standard deviation 
    us_importance = rf_us.impact()
    us_std = rf_us.std()
    
    # Call the plot to display feature importance
    feature_importance(usAB,us_importance, us_std)

    # Calculate NYC data and print results
    rf_ny = randforest()
    rf_ny_time = rf_ny.fitter(ny_x_train_std, ny_y_train)
    train_pred = rf_ny.predicter(ny_x_train_std)
    test_pred = rf_ny.predicter(ny_x_test_std)
    rf_ny_r2 = r2_score(ny_y_test, test_pred)
    print("\n\t-- ny Data --")
    print(f"Fitting Time:  {rf_ny_time:.2} seconds")
    print("Training R2:  ", r2_score(ny_y_train, train_pred))
    print("Training MSE: ", mean_squared_error(ny_y_train, train_pred))
    print("Testing R2:   ", r2_score(ny_y_test, test_pred))
    print("Testing MSE:  ", mean_squared_error(ny_y_test, test_pred))
    residual_plot(ny_y_train, ny_y_test, train_pred, test_pred, "Random Forest (NYC Data)")

    # Extract the feature importance and standard deviation 
    ny_importance = rf_ny.impact()
    ny_std = rf_ny.std()
    
    # Call the plot to display feature importance
    feature_importance(nyAB,ny_importance, ny_std)
    
    # Calculate CHI data and print results
    rf_ch = randforest()
    rf_ch_time = rf_ch.fitter(ch_x_train_std, ch_y_train)
    train_pred = rf_ch.predicter(ch_x_train_std)
    test_pred = rf_ch.predicter(ch_x_test_std)
    rf_ch_r2 = r2_score(ch_y_test, test_pred)
    print("\n\t-- ch Data --")
    print(f"Fitting Time:  {rf_ch_time:.2} seconds")
    print("Training R2:  ", r2_score(ch_y_train, train_pred))
    print("Training MSE: ", mean_squared_error(ch_y_train, train_pred))
    print("Testing R2:   ", r2_score(ch_y_test, test_pred))
    print("Testing MSE:  ", mean_squared_error(ch_y_test, test_pred))
    residual_plot(ch_y_train, ch_y_test, train_pred, test_pred, "Random Forest (CHI Data)")

    # Extract the feature importance and standard deviation 
    ch_importance = rf_ch.impact()
    ch_std = rf_ch.std()
    
    # Call the plot to display feature importance
    feature_importance(chAB, ch_importance, ch_std)
    
    # Plot comparison of fitting time and testing r2 score for each dataset
    fit_times = [rf_us_time, rf_ny_time, rf_ch_time]
    test_r2s = [rf_us_r2, rf_ny_r2, rf_ch_r2]
    datasets = ["US", "NY", "CH"]
    compare(fit_times, test_r2s, datasets)

if __name__ == '__main__':
    main() 

## CODE NO LONGER BEING USED ## 

# This is to check the new data set notice the rows are reduced
        
    # print("\n\t----- U.S. DATA -----\n")
    # print(usAB.isna().sum())
    # usAB.info()
    # print("Shape: ", usAB.shape)
            
    # print("\n\t----- N.Y.C. DATA -----\n")
    # print(nyAB.isna().sum())
    # nyAB.info()
    # print("Shape: ", nyAB.shape)

    # print("\n\t----- CHICAGO DATA -----\n")
    # print(chAB.isna().sum())
    # chAB.info()
    # print("Shape: ", chAB.shape)

# Because we previously used the NYC and Chicago datasets for testing,
# we used to recombine the training and testing into a single x and y dataframe

    # ny_x = pd.concat([ny_x_train, ny_x_test])
    # ny_y = pd.concat([ny_y_train, ny_y_test])
    # ch_x = pd.concat([ch_x_train, ch_x_test])
    # ch_y = pd.concat([ch_y_train, ch_y_test])

# Testing the random forest that was trained on US data on the other datasets

    # # Calculate NY data and print results
    # ny_pred = rf.predicter(ny_x)
    # print("\n\tNYC Data:")
    # print("Testing R2:   ", r2_score(ny_y, ny_pred))
    # print("Testing MSE:  ", mean_squared_error(ny_y, ny_pred))

    # # Calculate CH data and print results
    # ch_pred = rf.predicter(ch_x)
    # print("\n\tCH Data:")
    # print("Testing R2:   ", r2_score(ch_y, ch_pred))
    # print("Testing MSE:  ", mean_squared_error(ch_y, ch_pred))

# Linear Regression

    # print("\n--- LINEAR REGRESSION ---")
    # lr = linreg()
    # lr_time = lr.fitter(x_train, y_train)

    # # Calculate US data and print results
    # train_pred = lr.predicter(x_train)
    # test_pred = lr.predicter(x_test)
    # lr_r2 = r2_score(y_test, test_pred)
    # print("\n\tUS Data:")
    # print(f"Fitting Time:  {lr_time:.2} seconds")
    # print("Training R2:  ", r2_score(y_train, train_pred))
    # print("Training MSE: ", mean_squared_error(y_train, train_pred))
    # print("Testing R2:   ", r2_score(y_test, test_pred))
    # print("Testing MSE:  ", mean_squared_error(y_test, test_pred))
    # residual_plot(y_train, y_test, train_pred, test_pred, "Linear Regression (US Data)")

    # # Calculate NY data and print results
    # ny_pred = lr.predicter(ny_x)
    # print("\n\tNYC Data:")
    # print("Testing R2:   ", r2_score(ny_y, ny_pred))
    # print("Testing MSE:  ", mean_squared_error(ny_y, ny_pred))

    # # Calculate CH data and print results
    # ch_pred = lr.predicter(ch_x)
    # print("\n\tCH Data:")
    # print("Testing R2:   ", r2_score(ch_y, ch_pred))
    # print("Testing MSE:  ", mean_squared_error(ch_y, ch_pred))

# SVR

    # print("\n--- SVR ---")
    # svr = sv_reg()
    # svr_time = svr.fitter(x_train, y_train)

    # # Calculate US data and print results
    # train_pred = svr.predicter(x_train)
    # test_pred = svr.predicter(x_test)
    # svr_r2 = r2_score(y_test, test_pred)
    # print("\n\tUS Data:")
    # print(f"Fitting Time:  {svr_time:.2} seconds")
    # print("Training R2:  ", r2_score(y_train, train_pred))
    # print("Training MSE: ", mean_squared_error(y_train, train_pred))
    # print("Testing R2:   ", r2_score(y_test, test_pred))
    # print("Testing MSE:  ", mean_squared_error(y_test, test_pred))
    # residual_plot(y_train, y_test, train_pred, test_pred, "SVR (US Data)")

    # # Calculate NY data and print results
    # ny_pred = svr.predicter(ny_x)
    # print("\n\tNYC Data:")
    # print("Testing R2:   ", r2_score(ny_y, ny_pred))
    # print("Testing MSE:  ", mean_squared_error(ny_y, ny_pred))

    # # Calculate CH data and print results
    # ch_pred = svr.predicter(ch_x)
    # print("\n\tCH Data:")
    # print("Testing R2:   ", r2_score(ch_y, ch_pred))
    # print("Testing MSE:  ", mean_squared_error(ch_y, ch_pred))

# Elastic Net

    # print("\n--- ELASTIC NET ---")
    # elnet = en()
    # elnet_time = elnet.fitter(x_train, y_train)

    # # Calculate US data and print results
    # train_pred = elnet.predicter(x_train)
    # test_pred = elnet.predicter(x_test)
    # elnet_r2 = r2_score(y_test, test_pred)
    # print("\n\tUS Data:")
    # print(f"Fitting Time:  {elnet_time:.2} seconds")
    # print("Training R2:  ", r2_score(y_train, train_pred))
    # print("Training MSE: ", mean_squared_error(y_train, train_pred))
    # print("Testing R2:   ", r2_score(y_test, test_pred))
    # print("Testing MSE:  ", mean_squared_error(y_test, test_pred))
    # residual_plot(y_train, y_test, train_pred, test_pred, "Elastic Net (US Data)")

    # # Calculate NY data and print results
    # ny_pred = elnet.predicter(ny_x)
    # print("\n\tNYC Data:")
    # print("Testing R2:   ", r2_score(ny_y, ny_pred))
    # print("Testing MSE:  ", mean_squared_error(ny_y, ny_pred))

    # # Calculate CH data and print results
    # ch_pred = elnet.predicter(ch_x)
    # print("\n\tCH Data:")
    # print("Testing R2:   ", r2_score(ch_y, ch_pred))
    # print("Testing MSE:  ", mean_squared_error(ch_y, ch_pred))

# RANSAC Regressor

    # print("\n--- RANSAC ---")
    # rs = sac()
    # rs_time = rs.fitter(x_train, y_train)

    # # Calculate US data and print results
    # train_pred = rs.predicter(x_train)
    # test_pred = rs.predicter(x_test)
    # rs_r2 = r2_score(y_test, test_pred)
    # print("\n\tUS Data:")
    # print(f"Fitting Time:  {rs_time:.2} seconds")
    # print("Training R2:  ", r2_score(y_train, train_pred))
    # print("Training MSE: ", mean_squared_error(y_train, train_pred))
    # print("Testing R2:   ", r2_score(y_test, test_pred))
    # print("Testing MSE:  ", mean_squared_error(y_test, test_pred))
    # residual_plot(y_train, y_test, train_pred, test_pred, "RANSAC (US Data)")

    # # Calculate NY data and print results
    # ny_pred = rs.predicter(ny_x)
    # print("\n\tNYC Data:")
    # print("Testing R2:   ", r2_score(ny_y, ny_pred))
    # print("Testing MSE:  ", mean_squared_error(ny_y, ny_pred))

    # # Calculate CH data and print results
    # ch_pred = rs.predicter(ch_x)
    # print("\n\tCH Data:")
    # print("Testing R2:   ", r2_score(ch_y, ch_pred))
    # print("Testing MSE:  ", mean_squared_error(ch_y, ch_pred))

# This is a call to the PCA with all principle components returned. 
# We are no longer utilizing PCA and KPCA

    # x_train_pca,x_test_pca = pca(x_train_std,x_test_std)
    # print("\nPCA Analysis (On US Data Only):")

# Linear Regression

    # print("\n--- LINEAR REGRESSION ---")
    # lr = linreg()
    # lr_time = lr.fitter(x_train_pca, y_train)

    # # Calculate US data and print results
    # train_pred = lr.predicter(x_train_pca)
    # test_pred = lr.predicter(x_test_pca)
    # lr_r2 = r2_score(y_test, test_pred)
    # print("\n\tUS Data:")
    # print(f"Fitting Time:  {lr_time:.2} seconds")
    # print("Training R2:  ", r2_score(y_train, train_pred))
    # print("Training MSE: ", mean_squared_error(y_train, train_pred))
    # print("Testing R2:   ", r2_score(y_test, test_pred))
    # print("Testing MSE:  ", mean_squared_error(y_test, test_pred))
    # residual_plot(y_train, y_test, train_pred, test_pred, "Linear Regression (PCA)")

# SVR

    # print("\n--- SVR ---")
    # svr = sv_reg()
    # svr_time = svr.fitter(x_train_pca, y_train)

    # # Calculate US data and print results
    # train_pred = svr.predicter(x_train_pca)
    # test_pred = svr.predicter(x_test_pca)
    # svr_r2 = r2_score(y_test, test_pred)
    # print("\n\tUS Data:")
    # print(f"Fitting Time:  {svr_time:.2} seconds")
    # print("Training R2:  ", r2_score(y_train, train_pred))
    # print("Training MSE: ", mean_squared_error(y_train, train_pred))
    # print("Testing R2:   ", r2_score(y_test, test_pred))
    # print("Testing MSE:  ", mean_squared_error(y_test, test_pred))
    # residual_plot(y_train, y_test, train_pred, test_pred, "SVR (PCA)")

# Elastic Net

    # print("\n--- ELASTIC NET ---")
    # elnet = en()
    # elnet_time = elnet.fitter(x_train_pca, y_train)

    # # Calculate US data and print results
    # train_pred = elnet.predicter(x_train_pca)
    # test_pred = elnet.predicter(x_test_pca)
    # elnet_r2 = r2_score(y_test, test_pred)
    # print("\n\tUS Data:")
    # print(f"Fitting Time:  {elnet_time:.2} seconds")
    # print("Training R2:  ", r2_score(y_train, train_pred))
    # print("Training MSE: ", mean_squared_error(y_train, train_pred))
    # print("Testing R2:   ", r2_score(y_test, test_pred))
    # print("Testing MSE:  ", mean_squared_error(y_test, test_pred))
    # residual_plot(y_train, y_test, train_pred, test_pred, "Elastic Net (PCA)")

# Random Forest

    # print("\n--- RANDOM FOREST ---")
    # rf = randforest()
    # rf_time = rf.fitter(x_train_pca, y_train)


    # # Calculate US data and print results
    # train_pred = rf.predicter(x_train_pca)
    # test_pred = rf.predicter(x_test_pca)
    # rf_r2 = r2_score(y_test, test_pred)
    # print("\n\tUS Data:")
    # print(f"Fitting Time:  {rf_time:.2} seconds")
    # print("Training R2:  ", r2_score(y_train, train_pred))
    # print("Training MSE: ", mean_squared_error(y_train, train_pred))
    # print("Testing R2:   ", r2_score(y_test, test_pred))
    # print("Testing MSE:  ", mean_squared_error(y_test, test_pred))
    # residual_plot(y_train, y_test, train_pred, test_pred, "Random Forest (PCA)")


# RANSAC Regressor

    # print("\n--- RANSAC ---")
    # rs = sac()
    # rs_time = rs.fitter(x_train_pca, y_train)

    # # Calculate US data and print results
    # train_pred = rs.predicter(x_train_pca)
    # test_pred = rs.predicter(x_test_pca)
    # rs_r2 = r2_score(y_test, test_pred)
    # print("\n\tUS Data:")
    # print(f"Fitting Time: {rs_time:.2} seconds")
    # print("Training R2:  ", r2_score(y_train, train_pred))
    # print("Training MSE: ", mean_squared_error(y_train, train_pred))
    # print("Testing R2:   ", r2_score(y_test, test_pred))
    # print("Testing MSE:  ", mean_squared_error(y_test, test_pred))
    # residual_plot(y_train, y_test, train_pred, test_pred, "RANSAC (PCA)")

# Plot comparison of fitting time and US testing r2 score for each model
    # fit_times = [lr_time, svr_time, elnet_time, rf_time, rs_time]
    # test_r2s = [lr_r2, svr_r2, elnet_r2, rf_r2, rs_r2]
    # models = ["LinearRegression", "SVR", "ElasticNet", "RandomForest", "RANSAC"]
    # compare(fit_times, test_r2s, models)