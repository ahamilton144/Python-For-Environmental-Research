# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 22:19:56 2022

@author: rcuppari
"""

#################################################################
##   After finishing our timeseries analysis lecture, we are   ##
##        just going to work with some VERY messy data         ##
#################################################################

################
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
################

## read in the two overall csvs, call them 'means_transport' and 'demo_commuters'
## you will need a new argument here compared to usual: encoding="cp1252"
## just put a comma after the csv name and add this phrase before the 
## closing parenthesis. See the data cleaning file for more info 
means_transport = pd.read_csv("means_transport_overall.csv", encoding="cp1252")
demo_commuters = pd.read_csv("demo_commuters_overall.csv", encoding="cp1252")
means_transportF = pd.read_csv("means_transport_female.csv", encoding="cp1252")
means_transportM = pd.read_csv("means_transport_male.csv", encoding="cp1252")

## this function takes a dataframe (df) and a column index from which all 
## columns need to be made into numeric types, and will clean the data 
## by removing commas, %, and "N"s and finally convert it into a number 
## the default start_col is 2 because that is it for the datasets I have 
## provided 
def make_numeric(df, start_col = 2): 
    ## want to iterate over every single column and do some cleaning
#    print(df_new.head())
    for c in range(start_col, len(df.columns)): 
    ## take out commas and percent symbols
        df.iloc[:,c] = df.iloc[:,c].str.replace(',','')
        df.iloc[:,c] = df.iloc[:,c].str.replace('%','')
    ## set Ns as nan 
        df.iloc[:,c] = df.iloc[:,c].str.replace('N', 'na') 
    ## make the column numeric so we can subtract the two
        df.iloc[:,c] = pd.to_numeric(df.iloc[:,c], errors = 'coerce') ## force it to make errors nas
    return df

meansF_clean = make_numeric(means_transportF) 
meansM_clean = make_numeric(means_transportM)
demo_comm_clean = make_numeric(demo_commuters)
    
########################### IN-CLASS EXERCISE #################################
## I have cleaned this data so it is all numeric beyond column *2*
## subtract meansM_clean and meansF_clean to find the difference in the values
## between male and female responses (hint: this should be one line, but can be
## a for loop, and you will need to specify columns 2 onwards)


## THEN: add a column for the labels so you can see which is which 
## hint: we should be able to simply retrieve the "Label (Grouping)" column
## and add it back in 


## FINALLY: subset the data to retrieve row 20 -- Worked in a place of residence 
## let's call this "voi_F" and "voi_M" (variable of interest)

###############################################################################

## now let's figure out if there is a relationship... let's plot the two 
## with two different subplots 
dataF = meansF_clean.iloc[20,2:]
dataM = meansM_clean.iloc[20,2:]
dataF = pd.to_numeric(dataF)
dataM = pd.to_numeric(dataM)

## first set our standard parameters 
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 3
matplotlib.rcParams['font.size'] = 16

##
fig, (ax1, ax2) = plt.subplots(1, 2) ## number of rows, number of columns
fig.suptitle("Comparison of Male and Female Commuters", fontsize = 22)

## we plot the two subplots individually, specifying which plot we want our 
## lines to go on (ax1 or ax2). Essentially replace "plt" with the subplot label
ax1.plot(dataF, linewidth = 2, color = 'violet', label = 'Female')
ax1.plot(dataM, linewidth = 2, color = 'green', label = 'Male')
ax1.set_xlabel("Year")
ax1.set_ylabel("Percentage of Work From Home-rs")
ax1.set_xticks(['2010', '2012', '2014', '2016', '2018', '2020'], 
               ['2010', '2012', '2014', '2016', '2018', '2020'])
ax1.legend()

sns.regplot(ax = ax2, x = dataF, y = dataM)
ax2.set_xlabel("Percentage of Female WFH-rs")
ax2.set_ylabel("Percentage of Male WFH-rs")

## notice anything else different compared to our regular plots?
## why did I use strings for the x ticks?
## what's the deal with sns? 


########################### IN-CLASS EXERCISE #################################
##          What would you like to be able to do with this data?             ##








