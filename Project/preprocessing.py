# Imports for project
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA

## PREPROCESSING ##
 
# Load data by number
def load_data(num):
    d1 = "AB_US_2020.csv"
    d2 = "AB_NYC_2019.csv"
    d3 = "AB_CH_2021.csv"
    if num == 1:
        name = pd.read_csv(d1,low_memory=False)
    elif num == 2:
        name = pd.read_csv(d2)
    else:
        name = pd.read_csv(d3) 
    return name

# Cleaning data  
def get_clean_data(data):
    
    #if 'Chicago Lawn' in data['neighbourhood'].unique():
    data = data.drop(columns=("neighbourhood_group"),axis=1)
        
    data['price_log'] = np.log10(data['price'])

    # Drop -infinite values
    data.drop(data[data['price_log'] < 0].index, inplace = True)

        
    le = LabelEncoder()
    
    data = data.dropna()
    data = data.replace(np.nan,0)
    
    data['reviews_per_month'] = data['reviews_per_month'].astype(int)
    data['last_review'] = pd.to_datetime(data['last_review']).astype('int64')

    data["room_type"]=le.fit_transform(data["room_type"])
    data['name']=le.fit_transform(data['name'])
    data['host_name']=le.fit_transform(data['host_name'])
    
    if 'neighbourhood_group' in data.columns:
        data['neighbourhood_group']=le.fit_transform(data['neighbourhood_group'])
        
    data['neighbourhood']=le.fit_transform(data['neighbourhood'])
    
    if 'city' in data.columns:
        data['city']=le.fit_transform(data['city'])
            
    return data

def split_data(data):
    X = data.drop(columns=['price','price_log'],axis=1)
    y = data['price_log']
                      
    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=.3,random_state=1)
    
    return x_train,x_test,y_train,y_test

# Standard scaling the data
def std_data(x1,x2):
    sc = StandardScaler()
    x1_std = sc.fit_transform(x1)
    x2_std = sc.transform(x2)
    return x1_std,x2_std

# Principal component analysis
def pca(x1,x2):
    pca = PCA(n_components=3)
    x_train_pca = pca.fit_transform(x1)
    x_test_pca = pca.transform(x2)
    return x_train_pca,x_test_pca

def kpca(x1,x2):
    kpca = KernelPCA(n_components=3,kernel='rbf',gamma=1)
    x_train_kpca = kpca.fit_transform(x1)
    x_test_kpca = kpca.transform(x2)
    return x_train_kpca,x_test_kpca

def main(): 
    
# Load in the data for each dataset
# must provide name for data set and 1 for the US dataset or 2 for the second data set
    usAB = load_data(1)
    nyAB = load_data(2)
    chAB = load_data(3)
       
# This function converts the dataset to all int/float or date data types and removes missing values 
    usAB = get_clean_data(usAB)
    nyAB = get_clean_data(nyAB)
    chAB = get_clean_data(chAB)
   
# This is to check the new data set notice the rows are reduced

    print("\n\t----- U.S. DATA -----\n")
    # print(usAB.isna().sum())
    usAB.info()
    print("Shape: ", usAB.shape)
            
    print("\n\t----- N.Y.C. DATA -----\n")
    # print(nyAB.isna().sum())
    nyAB.info()
    print("Shape: ", nyAB.shape)

    print("\n\t----- CHICAGO DATA -----\n")
    # print(chAB.isna().sum())
    chAB.info()
    print("Shape: ", chAB.shape)
        
# This function will split the data and return unscaled data
# just input the data set you want to use in the function and make sure you have the number of variables to store the return into

    x_train,x_test,y_train,y_test = split_data(usAB)
    x_train_nyc,x_test_nyc,y_train_nyc,y_test_nyc = split_data(nyAB)
    x_train_chi,x_test_chi,y_train_chi,y_test_chi = split_data(chAB)
    
# standardized the data using standard scaler
# need two variables to store the returning data

    x_train_std, x_test_std = std_data(x_train,x_test)
    x_train_nyc_std, x_test_nyc_std = std_data(x_train_nyc,x_test_nyc)
    x_train_chi_std, x_test_chi_std = std_data(x_train_chi,x_test_chi)
    
# This is a call to the PCA with all principle components returned. 
# call this function with the x_train and x_test 
# returns to the two variables

    x_train_pca,x_test_pca = pca(x_train_std,x_test_std)
    x_train_nyc_pca,x_test_nyc_pca = pca(x_train_nyc_std,x_test_nyc_std)
    x_train_chi_pca,x_test_chi_pca = pca(x_train_chi_std,x_test_chi_std)
    
# This is a call to the KPCA with all principle components returned. 
# call this function with the x_train and x_test 
# returns to the two variables    

    x_train_kpca,x_test_kpca = kpca(x_train_std,x_test_std)
    # US data set throws an error because it is too large
    
    x_train_nyc_kpca,x_test_nyc_kpca = kpca(x_train_nyc_std,x_test_nyc_std)
    x_train_chi_kpca,x_test_chi_kpca = kpca(x_train_chi_std,x_test_chi_std)
    
if __name__ == '__main__':
    main() 
