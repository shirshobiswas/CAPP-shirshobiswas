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
    
      

#Wrapper function to print summary stats and plot histograms
def summary_wrapper(given_data, output_file):
    data = read_data(given_data)
    fields = list(data.columns.values)
    output = summary_stats(data,fields)
    
    with open(output_file, "w") as a:
        for x in output:
            print(x,file = a)
            
    #Need histograms only for numeric fields  
    data['Age'] = data['Age'].fillna(14)
    figure = data['Age'].hist()
    figure.set_title('Age')
    plt.draw()
    plt.savefig('Age')
    plt.close()
    
            
            
            

    
    
#Calling Functions    
summary_wrapper('mock_student_data.csv','summary_stats.txt')
  
 
    