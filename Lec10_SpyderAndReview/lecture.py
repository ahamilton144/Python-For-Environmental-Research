# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 10:35:59 2021

@author: rcuppari
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import os

names = ['AgLand','CO2','Electricity','Freshwater','GDP','Population','Renewables',
        'Traffic_Mortality','UrbanPop','WaterSan','MaternalMortality','Literacy']

df = pd.DataFrame()

for n in names:
    new = pd.read_csv(n+'.csv',skiprows=3)
    df = pd.concat([df,new])

df = df.iloc[:,:65]


countries = ['Afghanistan','Lebanon','Colombia','Brazil','Rwanda','Niger','Nepal','Bangladesh']

def country_plots(df, countries, variable): 
    ## Start by isolate our indicator variable 
    indicator = df[df['Indicator Name']==variable]
    
    ## write a line to subset the data based on whether the country column contains any of the 
    ## names within our list. There are a few ways to do this, but in order for the function to be useful, 
    ## you will probably want to find the number of values in the countries list and iterate over them
    
    df_subset = pd.DataFrame()
    for n in countries: 
        df_new = indicator[indicator['Country Name']==n]    
        ## For our purposes later, you will want to TRANSPOSE the data, that is flip it so that the 
        ## years are all in the first column. I have added the line below, meant to happen AFTER you subset the df
        ## I have also named your column 
        df_transposed = df_new.T
        df_transposed.columns = [str(n)] 
        ## you will also need to drop the first four rows (in your spare time, check out what happens if you do not)
        df_transposed = df_tranposed.iloc[5:,0]
        
        df_subset = pd.concat([df_subset,df_transposed],axis=1)
        ## this will be your combine dataset with year as index and columns as country 
        ## hint: concatenate the old with the new 

    ## create a line plot with different lines for each country over time
    ## hint: you can do this a few different ways, but a for loop is a safe bet (feel free to play around with this)
    
    ## also create a boxplot where the x axis is the year and the y axis is the spread for all of the countries combined 
    
    return df_subset

country_plot(df,countries,variable = 'GDP (current US$)')