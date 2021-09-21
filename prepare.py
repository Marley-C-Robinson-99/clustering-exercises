###################################         IMPORTS         ###################################
import pandas as pd
import os

from datetime import date
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

###################################         PREPARE FUNCS         ###################################
def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df


###################################         PREPARE/SPLIT         ###################################

def tvt_split(df, target, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for X_y spit if one is provided), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 


    The function returns, in this order, train, validate and test dataframes. 
    Or, if target is provided, in this order, X_train, X_validate, X_test, y_train, y_validate, y_test
    '''
    if target == None:
        X = df[[col for col in df.drop(columns = ['taxamount', 'tax_rate']).columns if df[col].dtype != object]]
        train_validate, test = train_test_split(X, test_size=0.2, 
                                                random_state=seed)
        train, validate = train_test_split(train_validate, test_size=0.3, 
                                                random_state=seed)
        return train, validate, test
    else:
        X = df[[col for col in df.drop(columns = ['taxamount', 'tax_rate']).columns if df[col].dtype != object]].drop(columns = target)
        y = df[target]
        X_train_validate, X_test, y_train_validate, y_test  = train_test_split(X, y, 
                                    test_size=0.2, 
                                    random_state=seed)
        X_train, X_validate, y_train, y_validate  = train_test_split(X_train_validate, y_train_validate, 
                                    test_size=.3, 
                                    random_state=seed)
        return X_train, X_validate, X_test, y_train, y_validate, y_test
