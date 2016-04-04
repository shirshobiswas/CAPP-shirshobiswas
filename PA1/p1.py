#In the beginning I define the required functions and in the end I call them

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import urllib.request

#Function to read the data
def read_data(given_data):
    data=pd.read_csv(given_data,index_col = 'ID', na_values = [''])
    return data

#Function to generate the summary stats
#Need mean,sd and median only for numeric data

#Part 1
def summary_stats(data,fields):
    output = list()
    for i in fields:
        output.append("Field Name: " + i)
        if data[i].dtype == 'float64':
            output.append("Mean: " + str(data[i].mean()))
            output.append("Std Dev: " +str(data[i].std()))
            output.append("Median: " + str(data[i].median()))
        output.append("Mode: " + str(data[i].mode()))
        output.append('\n')
        output.append('\n')
        
    return output
    
#Function to plot histogram separately for each field
def plot_hist(data,field):
    figure = data[field].hist()
    figure.set_title(field)
    plt.draw()
    plt.savefig(field)
    plt.close()

#Wrapper function to print summary stats and plot histograms
def summary_wrapper(given_data, output_file):
    data = read_data(given_data)
    fields = list(data.columns.values)
    output = summary_stats(data,fields)
    
    with open(output_file, "w") as a:
        for x in output:
            print(x,file = a)
            
    #Need histograms only for numeric fields        
    for field in fields:
        if data[field].dtype == 'float64':
            plot_hist(data,field)
            
            
    #If we want histograms to include missing values, we need to fill the missing with some number and then plot the histogram
    
    #For Age, since all values are above 15, fill missing with 14
    data['Age'] = data['Age'].fillna(14)
    figure = data['Age'].hist()
    figure.set_title('Age_2')
    plt.draw()
    plt.savefig('Age_2')
    plt.close()
    
    #For GPA, since all values are above 2, fill missing with 1
    data['GPA'] = data['GPA'].fillna(1)
    figure = data['GPA'].hist()
    figure.set_title('GPA_2')
    plt.draw()
    plt.savefig('GPA_2')
    plt.close()   
    
    
    #For Days_missed, since all values are above 0, fill missing with -1
    data['Days_missed'] = data['Days_missed'].fillna(-1)
    figure = data['Days_missed'].hist()
    figure.set_title('Days_missed_2')
    plt.draw()
    plt.savefig('Days_missed_2')
    plt.close() 

        #Next we plot the histograms for Gender and Graduated and State. I show two ways of generating histograms for string valued data
     
    
    data['Gender'] = data['Gender'].fillna('Missing')
    data.Gender.value_counts()
    s = pd.Series({'Female':398,'Male':376, 'Missing':226})
    figure = s.plot(kind = 'bar', rot = 0)
    figure.set_title('Gender')
    plt.draw()
    plt.savefig('Gender')
    plt.close() 
                  
    
                  
    data.Graduated.value_counts()
    s = pd.Series({'Yes':593,'No':407})
    figure = s.plot(kind = 'bar', rot = 0)
    figure.set_title('Graduated')
    plt.draw()
    plt.savefig('Graduated')
    plt.close() 
                 
    figure = data.State.value_counts().plot(kind = 'bar')
    plt.draw()
    plt.savefig('State')
    plt.close

    #summary of missing data
    df.isnull().sum()

 #Part 2
    
def genderize_api(first_name):
    webservice_url = "https://api.genderize.io/?name=" + first_name
    gender_data = json.loads(urllib.request.urlopen(webservice_url).read().decode("utf8"))
    return gender_data['gender']
 
    

def genderize_wrapper(given_data):
    data = read_data(given_data)
    gender_missing = data[data['Gender'].isnull()]['First_name']
    for name in gender_missing:
        gender = genderize_api(name)
        data.loc[(data['Gender'].isnull()) & (data['First_name'] == name), 'Gender'] = gender
        
        
    data.to_csv('missing_gender_filled.csv')
    
 #Part 3
    
def fill_missing_with_mean(data, given_data, output_file, fields):
    for field in fields:
        mean = data[field].mean()
        data[field] = data[field].fillna(mean)
        
    data = data.round({'Age':0, 'GPA':0, 'Days_missed':0})
    data.to_csv(output_file)
    
def fill_missing_with_conditional_mean(data,given_data,output_file, grouping_conditions, fields):  
    for field in fields:
        mean = data.groupby(grouping_conditions)[field].mean()
        data = data.set_index(grouping_conditions)
        data[field] = data[field].fillna(mean)
        data = data.reset_index()
        
    data = data.round({'Age':0, 'GPA':0,'Days_missed':0})
    data.to_csv(output_file)
    
def fill_missing_wrapper(given_data):
    data = read_data(given_data)
    fields = ["Age", "GPA", "Days_missed"]
    
    fill_missing_with_mean(data, given_data, 'missing_with_mean.csv',fields)
    fill_missing_with_conditional_mean(data, given_data, 'missing_with_conditional_mean_a.csv', ['Graduated'], fields)
    fill_missing_with_conditional_mean(data, given_data, 'missing_with_conditional_mean_b.csv', ['Graduated', 'Gender'], fields)
    
    
    
def drop_missing_values(given_data):
    data = read_data(given_data)
    data = data[(data['Age'].notnull()) & (data['GPA'].notnull()) & (data['Days_missed'].notnull())]
    data.to_csv('drop_missing_values.csv')
    
    
    
#Calling Functions    
summary_wrapper('mock_student_data.csv','summary_stats.txt')
genderize_wrapper('mock_student_data.csv')
fill_missing_wrapper('missing_gender_filled.csv')
drop_missing_values('mock_student_data.csv')   
 
    
    
    




         
            
