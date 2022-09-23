# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 16:49:21 2022

@author: rcuppari
"""

import pandas as pd
import numpy as np
import random 

#############################################################################
################# LECTURE 6: PANDAS AND REVIEW ######################
#############################################################################

## GOALS: familiarize ourselves with the Pandas package + do cool things with it
## truly my favorite package of them all, mostly because I love dataframes 
## to me, dfs (DataFrames) are the Excel of Python

## I hope to have time to also introduce very basic plotting with matplotlib
#############################################################################
 

## to start -- my qualitative q for hw4
#Creating a single instance of storage,forecast and season 
def res_releases(storage, forecast, season):
    if storage == 'flood':
        if forecast == 'wet': 
            release = 1000 
        else:
            release = 700
            
    elif storage == 'normal': 
        if forecast =='wet': 
            release = 600
        else: 
            release = 350
        
    else: 
        if forecast == 'wet':
            if (season == 'winter') or (season == 'spring'): 
                release = 300
            else: 
                release = 200
        else: 
            if (season == 'winter') or (season == 'spring'): 
                release = 300
            else: 
                release = 100
    
    return release

##Random list for iteration
seasons = random.choices((['spring', 'summer', 'fall', 'winter']), weights = [0.25, 0.25, 0.25, 0.25], k = 100)
print(seasons[:10])

storage = random.choices((['normal', 'drought', 'flood']), weights = [0.5, 0.25, 0.25], k = 100)
print(storage[:10])

forecast = random.choices((['wet', 'dry']), weights = [0.7, 0.3], k = 100)
print(forecast[:10])

releases = []

for index in range(len(forecast)):
    release = res_releases(storage=storage[index], forecast=forecast[index], season=seasons[index])
    releases.append(release)

## THE ALTERNATE CASE: now storage is the full timeseries 
def res_releases_mult(storage, forecast, season):
    release = []
    for index in range(len(storage)):     
        if storage[index] == 'flood':
            if forecast[index] == 'wet': 
                new_release = 1000 
            else:
                new_release = 700
                
        elif storage[index] == 'normal': 
            if forecast[index] =='wet': 
                new_release = 600
            else: 
                new_release = 350
            
        else: 
            if forecast[index] == 'wet':
                if (season[index] == 'winter') or (season[index] == 'spring'): 
                    new_release = 300
                else: 
                    new_release = 200
            else: 
                if (season[index] == 'winter') or (season[index] == 'spring'): 
                    new_release = 300
                else: 
                    new_release = 100
        release.append(new_release)
    return release

release2 = res_releases_mult(storage, forecast, seasons)

print(releases)
print(release2)

############################### PANDAS ########################################
## we can create dataframes much like dictionaries 
## numeric index for each household in county
household = ['H' + str(i) for i in list(range(1000))]
print(household[:20])

pop_ch = 61912
pop_ra = 467665.
pop_dur = 276341

frac_ch = pop_ch / (pop_ch + pop_ra + pop_dur)
frac_ra = pop_ra / (pop_ch + pop_ra + pop_dur)
frac_dur = pop_dur / (pop_ch + pop_ra + pop_dur)

city = [random.choices(['Durham', 'Chapel Hill', "Raleigh"], 
                       weights = [frac_dur, frac_ch, frac_ra], 
                       k=1)[0] for h in household]
print(city[:20])

## state for each household
nc = ['nc'] * 1000

## we create dfs the same way we make dictionaries
df = pd.DataFrame({'household': household, 'state': nc, 'city': city})

## we can also specify an index! 
df2 = pd.DataFrame({'household': household, 'state': nc, 'city': city}, index = household)

## let's assign income using a for loop and LABEL INDEXING
df['income'] = 0 
for row in range(len(df)): 
    if df.loc[row, 'city'] == 'Durham': 
         df.loc[row, 'income'] = random.gauss(35164, 20000)
    elif df.loc[row, 'city'] == 'Raleigh': 
         df.loc[row, 'income'] = random.gauss(37393, 22000)
    else: 
        df.loc[row, 'income'] = random.gauss(24670, 24000)
         
print(df)
## NOTE: we can have different object types! E.g., float and string

## we can also index using the INTEGER location, "iloc"
df2['income'] = 0 
for row in range(len(df2)): 
    if df2.iloc[row, 2] == 'Durham': 
         df2.iloc[row, 3] = random.gauss(35164, 20000)
    elif df2.iloc[row, 2] == 'Raleigh': 
         df2.iloc[row, 3] = random.gauss(37393, 22000)
    else: 
        df2.iloc[row, 3] = random.gauss(24670, 24000)
         
print(df2)

## we can use either of these to get slices of our data 
print(df2.iloc[:4, 3])

## what would we want to know about our df? Columns, shape? 
## what would happen if we used ".loc" on df2?

## we can always reset our index if it's confusing 
df2.reset_index(inplace = True) 

################# READING IN DATA!!!!! #################
## we use pandas! Either "pd.read_csv()" or "pd.read_excel()
co2 = pd.read_csv("Data/CO2.csv")

## didn't work -- let's open up our data and see why 
## use "header = x" or "skip = "x" to ignore the first x lines or choose the line
## with column names 
co2 = pd.read_csv("Data/CO2.csv", header = 2)
co2 = pd.read_csv("Data/CO2.csv", skiprows = 4)

## and you can preview your df
co2.head()
co2.tail()
co2.columns

## and we can use a few different tools to add columns
## INCLUDING using numpy! 
nrow = co2.shape[0]
print(nrow)
random = np.random.choice([1, 2, 3, 4, 5, 6], size = nrow, p = [0.3, 0.25, 0.15, 0.15, 0.075, 0.075])
co2['value'] = random
co2.head()

## can preview based on our column name, and we can actually call columns in two ways
print(co2['value'].head())
co2.value.tail()

## you can also set your index to different values, which can be useful
co2.set_index('Country Name', inplace = True)

## we can also do math really easily with dfs... for example, what if we wanted 
## to convert metric tons per capita to lbs per capita just for the 2015 in a new df? 
new_co2 = co2[['Country Code', '2015']]
new_co2['lbs'] = co2['2015'] * 2204.62

## we might also want to retrieve summary stats -- easy! Similar to numpy
co2_stats = pd.DataFrame({'mean': new_co2.lbs.mean(), 
                          'min': new_co2.lbs.min(), 
                          'max': new_co2.lbs.max(), 
                          'med': new_co2.lbs.median(),
                          'std': new_co2.lbs.std(),
                          'q05': new_co2['lbs'].quantile(0.05),
                          'q95': new_co2.lbs.quantile(0.95)}, index = ['stat_value'])

## we can also pull a single column, but then our df becomes a SERIES
co2_2015 = co2['2015']

## now we index using just an integer, like an array 
print(co2_2015[1])

## we can also just make it an array 
co2_array = co2_2015.values

## finally, we can also use logical indexing for dfs just like we did for np arrays
is_belize = co2['Country Code'] == 'BLZ'
co2_belize = co2.loc[is_belize, :]

## another way to subset the data: 
co2_belize2 = co2[co2['Country Code'] == 'BLZ']
co2_blz_alb = co2[(co2['Country Code'] == 'BLZ') | (co2['Country Code'] == 'ALB')]
co2_not_blz_and_alb = co2[(co2['Country Code'] != 'BLZ') & (co2['Country Code'] != 'ALB')]

################## IN-CLASS EXERCISE ####################
## I have uploaded a bunch of global World Bank data for us to play around with
## similar to our first. We are going to try to automate the reading in process
## Create a for loop which takes the names of the indicators and reads in the csv files for each one, 
## concatenating them into one large dataframe

## Start by creating a list with the names of each data file in the folder 
names = ['AgLand','CO2','Electricity','Freshwater','GDP','Population','Poverty','Renewables',
        'Traffic_Mortality','UrbanPop','WaterSan','MaternalMortality','Literacy']

## create an empty dataframe to hold all of our information 
df = pd.DataFrame()


## Set up what the for loop is going to iterate over 
for n in names: 
    ## Within the for loop, write a line to read in the file where the file name is changing according to 
    ## the indicator. Skip 3 lines.
    ## hint: you will need to combine a non-string with the indicator name with a string ('.csv')
    new = pd.read_csv(##INSERT NAME!)
    ## Within the for loop, concatenate your files based on rows. 
    ## hint: pd.concat([df1,df2]) will concatenate based on rows 
    ## the paramater axis=0 is implied, changing this to  axis=1 will concatenate based on columns.
    df = 

df = df.iloc[:,:65] ##need to drop last column to clean data 


################## IN-CLASS EXERCISE ####################
## let's find the total traffic mortality given the per 100,000 stat and the population

