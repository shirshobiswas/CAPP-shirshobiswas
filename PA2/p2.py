import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
import sys
import random
from sklearn.linear_model import LogisticRegression
from __future__ import division

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category = DeprecationWarning)

#Evaluation metrics is a separate pyhton file containing some useful functions for evaluation of different methods. One of the function is evaluating a binary classifier
from EvaluationMetrics import bin_classif_eval

#This function reads in a csv file as a dataframe
def readcsvfile(file_name):
    return pd.read_csv(file_name, header = 0)

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
    dummies = pd.get_dummies(df['Category']).rename(columns = lambda x: 'Category_'=str(x))
    df = pd.concat([df,dummies],axis =1)
    df = df.drop(['Category'], inplace = True, axis = 1)
    
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


#This function fills the missing values in a list of columns in a datafraframe with the mean values of the column

def fill_missing_with_mean(df,col):
   
        df[col] = df[col].fillna(df[col].mean())
        
        
#This is where the code for this particular problem starts. All the functions above can directly be imported for any other ML problem as well

cs = readcsvfile('cs-training.csv')
summary_statistics(cs)
data_histogram(cs)

fill_missing_with_mean(cs,['NumberOfDependents'])
fill_missing_with_mean(cs,['MonthlyIncome'])


age_bins = [0] + range(20,80,5) + [120]
age_bucket = discretize_bins_values(cs, 'age', age_bins)

income_bins = range(0,10000,1000) + [cs['MonthlyIncome'].max()]
income_bucket = discretize_bins_values(cs,'MonthlyIncome', income_bins)

plot_by_group_mean(cs, ['NumberOfDependents', 'SeriousDlqin2yrs'], 'NumberOfDependents')
plot_by_group_mean(cs, [age_bucket, 'SeriousDlqin2yrs'], age_bucket)
plot_by_group_mean(cs, [income_bucket, 'SeriousDlqin2yrs'], income_bucket)

print sum(cs.SeriousDlqin2yrs) / len(cs)

cs_train, cs_valid = train_test_split(
    cs,
    train_size=.8,
    random_state=99)

log_reg_model = LogisticRegression(
    penalty='l2',
    dual=False,
    tol=0.0001,
    C=1000.,
    fit_intercept=True,
    intercept_scaling=1.,
    class_weight=None,
    random_state=99,
    solver='lbfgs',
    max_iter=100,
    multi_class='multinomial',
    verbose=0)

y_var_name = 'SeriousDlqin2yrs'
X_var_names = ['RevolvingUtilizationOfUnsecuredLines',  
  'age',
  'NumberOfTime30-59DaysPastDueNotWorse',
  'DebtRatio',
  'MonthlyIncome',
  'NumberOfOpenCreditLinesAndLoans',
  'NumberOfTimes90DaysLate',
  'NumberRealEstateLoansOrLines',
  'NumberOfTime60-89DaysPastDueNotWorse',
  'NumberOfDependents']

prob_threshold = 0.5

log_reg_model.fit(X=cs_train[X_var_names], y=cs_train.SeriousDlqin2yrs)
log_reg_pred_probs = log_reg_model.predict_proba(X=cs_valid[X_var_names])

log_reg_oos_performance = bin_classif_eval(
    log_reg_pred_probs[:, 1], cs_valid.SeriousDlqin2yrs,
    pos_cat=1, thresholds=prob_threshold)

prob_threshold2 = 0.05
log_reg_oos_performance = bin_classif_eval(
    log_reg_pred_probs[:, 1], cs_valid.SeriousDlqin2yrs,
    pos_cat=1, thresholds=prob_threshold2)