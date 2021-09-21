###################################         IMPORTS         ###################################
import pandas as pd
import os
from env import host, user, password

from datetime import date
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
###################################         ACQUIRE         ###################################
# Function to establish connection with Codeups MySQL server, drawing username, password, and host from env.py file
def get_db_url(host = host, user = user, password = password, db = 'zillow'):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

# Function to acquire neccessary zillow data from Codeup's MySQL server
def acquire_zillow():
    filename = "raw_zillow.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        SQL = '''
        SELECT *
        FROM properties_2017
        LEFT OUTER JOIN airconditioningtype 
            ON airconditioningtype.airconditioningtypeid = properties_2017.airconditioningtypeid
        LEFT OUTER JOIN architecturalstyletype
            ON architecturalstyletype.architecturalstyletypeid = properties_2017.architecturalstyletypeid
        LEFT OUTER JOIN buildingclasstype 
            ON buildingclasstype.buildingclasstypeid = properties_2017.buildingclasstypeid
        LEFT OUTER JOIN heatingorsystemtype
            ON heatingorsystemtype.heatingorsystemtypeid = properties_2017.heatingorsystemtypeid
        LEFT OUTER JOIN predictions_2017
            ON predictions_2017.id = properties_2017.id
        INNER JOIN (
            SELECT id, MAX(transactiondate) as last_trans_date 
            FROM predictions_2017
            GROUP BY id
            ) predictions ON predictions.id = properties_2017.id AND predictions_2017.transactiondate = predictions.last_trans_date
        LEFT OUTER JOIN propertylandusetype
            ON propertylandusetype.propertylandusetypeid = properties_2017.propertylandusetypeid
        LEFT OUTER JOIN storytype
            ON storytype.storytypeid = properties_2017.storytypeid
        LEFT OUTER JOIN typeconstructiontype
            ON typeconstructiontype.typeconstructiontypeid = properties_2017.typeconstructiontypeid
        JOIN unique_properties
            ON unique_properties.parcelid = properties_2017.parcelid
        WHERE latitude IS NOT NULL and longitude IS NOT NULL
        '''
        # read the SQL query into a dataframe
        df = pd.read_sql(SQL, get_db_url())

        # dropping duplicate fields
        df = df.loc[:,~df.columns.duplicated()]

        # renaming cols
        df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                              'bathroomcnt':'bathrooms', 
                              'calculatedfinishedsquarefeet':'area',
                              'taxvaluedollarcnt':'tax_value', 
                              'yearbuilt':'year_built',
                              'transactiondate':'sale_date'})


        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index=False)
        # Return the dataframe to the calling code
        return df

###################################         BASIC SUMMARY         ###################################
def object_vals(df):
    '''
    This is a helper function for viewing the value_counts for object cols.
    '''
    for col, vals in df.iteritems():
        print(df[col].value_counts(dropna = False))
        print('----------------------')

def col_desc(df):
    stats_df = df.describe().T
    stats_df['range'] = stats_df['max'] - stats_df['min']
    return stats_df

def null_cnts(df):
    for col in df.columns:
        print(f'{col}: {df[col].isna().sum()}')

def distributions(df):
    for col, vals in df.iteritems():
        if df[f'{col}'].dtype != object:
            print(sns.histplot(data = df[f'{col}']), plt.title(f'Distribution of {col}'),
            plt.show(), end = '\n------------------------------------\n')


def summarize_df(df):
    '''
    This function returns the shape, info, a preview, the value_counts of object columns
    and the summary stats for numeric columns.
    '''
    print(f'This dataframe has {df.shape[0]} rows and {df.shape[1]} columns.')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('****Info****')
    print(df.info())
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('****Null Counts****')
    null_cnts(df)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('****Value Counts****')
    object_vals(df)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('****Column Stats****')
    col_desc(df)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('****Distributions****')
    distributions(df)

