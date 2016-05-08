import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
import sys
import random
from __future__ import division
from numpy import nan

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category = DeprecationWarning)

#Evaluation metrics is a separate pyhton file containing some useful functions for evaluation of different methods. One of the function is evaluating a binary classifier
from EvaluationMetrics import bin_classif_eval

#This function reads in a csv file as a dataframe
def readcsvfile(file_name):
    df =  pd.read_csv(file_name, header = 0)
#We discussed in the previous TA session that NA values in some fields were coded numerically.
#I replace these numerical values with 'nan'

    df = df.replace({'NumberofTimes90DaysLate':{98:nan, 97:nan, 96:nan}, 'NumberofTime60-89DaysPastDueNotWorse':{98:nan, 97:nan, 96:nan}, 'NumberofTime30-59DaysPastDueNotWorse':{98:nan, 97:nan, 96:nan}, 'NumberofDependents':{20:nan}, 'age':{0:nan}})
    return df


#This function describes a datarame i.e. it prints column names, head and tail of the data,summary statistics, number of missing values in each column and the correlation matrix
def summary_statistics(df):
    pd.set_option('display.width', 18)
    print 'Column Names:', "\n", df.columns.values
    print 'First Few Rows of Data:', "\n", df.head()
    print 'Last Few Rows of Data:', "\n", df.tail()
    print 'Summary Statistics:', "\n", df.describe(include = 'all')
    print 'Number of Missing Values:', "\n", df.isnull().sum()
    
    for col_name in df:
        print ('Data Type %s: %s' %(col_name, df[col_name].dtype))
        
    print 'Correlation Matrix :', "\n", df.corr().unstack()
    
    
    
    
def plot_histogram(df, hist_var):
    fig = df[hist_var].hist()
    fig.set_title('Histogram for ' + hist_var)
    plt.draw()
    plt.savefig(hist_var)
    plt.close()


def plot_bar(df, bar_var):
    fig =df.groupby(bar_var).size().plot(kind='bar')
    fig.set_xlabel(bar_var) #defines the x axis label
    fig.set_ylabel('Number of Observations') #defines y axis label
    fig.set_title(bar_var+' Distribution') #defines graph title
    plt.draw()
    plt.savefig(bar_var)
    plt.close('all')
    
histogram_variables = ['serious_dlqin2yrs','revolving_utilization_of_unsecured_lines', 'age', 'number_of_time30-59_days_past_due_not_worse', 'debt_ratio', 'monthly_income', 'number_of_open_credit_lines_and_loans', 'number_of_times90_days_late', 'number_real_estate_loans_or_lines', 'number_of_time60-89_days_past_due_not_worse', 'number_of_dependents']





bar_variables = ['serious_dlqin2yrs']








#This function prints histograms for each column of a data frame
def data_histogram(df):
    df.hist()
    plt.savefig('histograms.png')
    
#This function plots grouped columns with mean of the group
def plot_by_group_mean(df,columns, group_by_col):
    df[columns].groupby(group_by_col).mean().plot()
    file_name = 'plot_by_' + group_by_col + '.png'
    plt.savefig(file_name)
    

#This function converts a categorical variable in a data frame into binary dummies and then drops the original categorical variable
def categorical_to_binary_dummies(df,Category):
    dummies = pd.get_dummies(df['Category'], Category, drop_first = True)
    df = df.join(dummies)
    return df
    
#This function takes a dataframe and a column name and discretizes a continuous variable into bins
def discretize_bins_values(df,col_name, bins, verbose = False):
    new_col = 'bins_' + str(col_name)
    df[new_col] = pd.cut(df[col_name], bins = bins, include_lowest = True, labels = False)
    
    if verbose:
        print pd.value_counts(data[new_col])
        
    return new_col

#This function takes a dataframe and a column name and discretizes a continuous variable into  bins based on quantiles
def discretize_bins_quantiles(df,col_name,number_of_bins, verbose = False):
    new_col = 'bins_' + str(col_name)
    df[new_col] = pd.qcut(df[col_name],number_of_bins, labels = False)
    
    if verbose:
        print pd.value_counts(data[new_col])
        
    return new_col


#This function returns the log of a column. useful to get log income
def log_column(df,col_name):
    log_col = 'log_' + str(col_name)
    df[log_col] = df[col_name].apply(lambda x: np.log(x+1))
    return log_col
#This function plots the histogram of a log variable

def plot_log(df,var):
    lb = 0
    ub = 15
    increment = 0.5
    plt.gca().set_xscale('log')
    fig = df[var].hist(bins = np.exp(np.arrange(lb,ub,increment)))
    fig.set_xlabel('log'+var)
    plt.savefig('log'+var)
    plt.close()
    

##Imputing Missing values in Training Data Set and Filling in Missing values in testing dataset with stored values in Testing Dataset

#This function fills the missing values in a column fn a datafraframe with mean, median or mode

def impute_missing_values(df,var,method):
   
        if method == 'mean':
               mean = df[var].mean()
               return mean

        elif method == 'median':
               median = df[var].median()
               return median

        elif method == 'mode':
               mode = df[var].mode[0]
               return mode


#This function fills the missing values in a column fn a datafraframe with a specified value

def replace_missing_values(df,var,value):
         df[var] = df[var].fillna(value)
         return df
        
        

