# Imports for project
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Imports from our other files
from preprocessing import *

## DATA VISUALIZATION ##

# Splits the data into subsample with an even distribution of specified feature
def get_subsample(data, feature, split):
    # Split the specified feature from the data
    x = data.loc[:, data.columns != feature]
    y = data[feature]
    subsample1, rest1, subsample2, rest2 = train_test_split(x,y,test_size=split)
    
    # Put the feature back before returning
    subsample1.insert(loc=data.columns.get_loc(feature),
          column=feature,
          value=subsample2)
    return subsample1

# Analyzes the prices associated with a given dataframe
def price_analysis(data, data_name, color):

    # Info about the prices
    print("Average Price: ", data["price"].mean())
    print("Minimum Price: ", data["price"].min())
    print("Maximum Price: ", data["price"].max())
    print("Median Price:  ", data["price"].median())

    # Plot the density of prices of the data
    fig,ax = plt.subplots(figsize=(8,8))
    title = 'Price Density for ' + data_name
    fig.suptitle(title)
    plt.xlim(0,1250)
    plt.xlabel('Price')
    plt.ylabel('Density')
    sns.kdeplot(data['price'],shade=True,color=color)
    plt.show()
    print()

# Generates various visualizations for our data
def data_vis(usAB, nyAB, chAB):

    # Pearson Corr
    us_corr = usAB.corr()
    ny_corr = nyAB.corr()
    ch_corr = chAB.corr()

    # Info about the prices
    print("\n\t----- U.S. DATA -----\n")
    price_analysis(usAB, "US Data", 'm')
    print("\n\t----- N.Y.C. DATA -----\n")
    price_analysis(nyAB, "NYC Data", 'y')
    print("\n\t----- CHICAGO DATA -----\n")
    price_analysis(chAB, "Chicago Data", 'c')

    # Plot the average price for each type of room
    fig,ax = plt.subplots(figsize=(8,8))
    fig.suptitle('Average Prices for each Room Type')
    sns.barplot(x="room_type",y="price", data=usAB)
    plt.show()
    print()

    # Plot the prices of the n most common cities
    n = 20
    cities = usAB['city'].value_counts()[:n].index.tolist()
    city_df = usAB.loc[usAB['city'].isin(cities)] 
    fig,ax = plt.subplots(figsize=(8,8))
    fig.suptitle("Prices by City")
    sns.barplot(x="price",y='city', data=city_df)
    plt.show()
 
 #Creates residual plot from given data and predictions
def residual_plot(y_train, y_test, y_train_pred, y_test_pred, title):
    plt.scatter(y_train_pred,y_train_pred - y_train, c='blue', marker='o', label='Training data')
    plt.scatter(y_test_pred, y_test_pred - y_test, c='green', marker='s', label='Test data')
    plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
    plt.legend(loc='best')
    plt.title(title)
    plt.show()
    
    
def corr_heat_map(data):
    if 'city' in data.columns:
        cols = ['id', 'name', 'host_id', 'host_name', 'neighbourhood', 'latitude',
               'longitude', 'room_type', 'price', 'minimum_nights',
               'number_of_reviews', 'last_review', 'reviews_per_month',
               'calculated_host_listings_count', 'availability_365', 'city','price_log']
    else:
        cols = ['id', 'name', 'host_id', 'host_name', 'neighbourhood', 'latitude',
               'longitude', 'room_type', 'price', 'minimum_nights',
               'number_of_reviews', 'last_review', 'reviews_per_month',
               'calculated_host_listings_count', 'availability_365','price_log']
    
    correlation_coefficient = np.corrcoef(data[cols].values.T)
    
    if 'city' in data.columns:
        fig, ax = plt.subplots(figsize=(15,8))
        ax.set_title("US Data Set")
        sns.heatmap(correlation_coefficient, annot=True, linewidths=.5,
        yticklabels = cols, xticklabels=cols)
        plt.show()
    elif (data['latitude'].values > 41).any():
        fig, ax = plt.subplots(figsize=(15,8))
        ax.set_title("Chi Data Set")
        sns.heatmap(correlation_coefficient, annot=True, linewidths=.5,
        yticklabels = cols, xticklabels=cols)
        plt.show()
    else:
        fig, ax = plt.subplots(figsize=(15,8))
        ax.set_title("NYC Data Set")
        sns.heatmap(correlation_coefficient, annot=True, linewidths=.5,
        yticklabels = cols, xticklabels=cols)
        plt.show()

def compare(fit_times, test_r2s, models):
    # Create figure to compare training and testing accuracy for each regressor
    plt.figure(figsize=(8,6))
    plt.scatter(fit_times,test_r2s)
    plt.xlabel("Fitting Times")
    plt.ylabel("Testing r2 Scores")
    plt.title("Time and Performance Comparison",fontsize=15)
    for i, label in enumerate(models):
        plt.annotate(label, (fit_times[i], test_r2s[i]))
    plt.show()


#creating a pairplot of a subset of each the US, NYC, AND CH 
# merged the dataset and also created it on a subset of each of them indivitually 
def pairplot(data):
    # Select columns based on the presence of the 'city' column
    if 'city' in data.columns:
        cols = ['neighbourhood', 'latitude',
               'longitude', 'room_type', 'minimum_nights',
               'number_of_reviews','reviews_per_month',
               'calculated_host_listings_count', 'availability_365', 'city','price_log']
    else:
        cols = ['neighbourhood', 'latitude',
               'longitude', 'room_type', 'minimum_nights',
               'number_of_reviews','reviews_per_month',
               'calculated_host_listings_count', 'availability_365','price_log']
     
    sns.pairplot(data[cols], diag_kind='kde', corner=True)
    plt.tight_layout()
    plt.show()

def pairplotmerge(usAB, nyAB, chAB):
    #merge into a single dataframe
    airbnb_df = pd.concat([usAB,nyAB,chAB],ignore_index=True)

    # Select columns based on the presence of the 'city' column
    if 'city' in airbnb_df.columns:
        cols = ['id', 'name', 'host_id', 'host_name', 'neighbourhood', 'latitude',
               'longitude', 'room_type', 'price', 'minimum_nights',
               'number_of_reviews', 'last_review', 'reviews_per_month',
               'calculated_host_listings_count', 'availability_365', 'city','price_log']
    else:
        cols = ['id', 'name', 'host_id', 'host_name', 'neighbourhood', 'latitude',
               'longitude', 'room_type', 'price', 'minimum_nights',
               'number_of_reviews', 'last_review', 'reviews_per_month',
               'calculated_host_listings_count', 'availability_365','price_log']
    


    # Create a pair plot of all three datasets combined
    sns.pairplot(airbnb_df, diag_kind='kde', hue='city', corner=True)
    
    # Create a pair plot for the merged dataset using all features
    sns.pairplot(airbnb_df, diag_kind='kde', corner=True)
    plt.tight_layout()
    plt.show()

def feature_importance(data,impact,std):
    if 'city' in data.columns:
        cols = ['id', 'name', 'host_id', 'host_name', 'neighbourhood', 'latitude',
               'longitude', 'room_type', 'minimum_nights',
               'number_of_reviews', 'last_review', 'reviews_per_month',
               'calculated_host_listings_count', 'availability_365', 'city']
    else:
        cols = ['id', 'name', 'host_id', 'host_name', 'neighbourhood', 'latitude',
               'longitude', 'room_type', 'minimum_nights',
               'number_of_reviews', 'last_review', 'reviews_per_month',
               'calculated_host_listings_count', 'availability_365']
        
    forest_importances = pd.Series(impact, index=cols)
    
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    
def main():
    # Import the data
    usAB = load_data(1)  
    nyAB = load_data(2)
    chAB = load_data(3)
    
    # Generate visualizations
    data_vis(usAB, nyAB, chAB)
    
    # clean data for correlation map
    usAB = get_clean_data(usAB)
    nyAB = get_clean_data(nyAB)
    chAB = get_clean_data(chAB)
    
    
    # correlation heat maps
    corr_heat_map(usAB)
    corr_heat_map(nyAB)
    corr_heat_map(chAB)

    #pairplots 
    pairplotmerge(usAB, nyAB, chAB) #merged 
    #individual
    pairplot(usAB)
    pairplot(nyAB)
    pairplot(chAB)




if __name__ == '__main__':
    main() 

    
    
 
