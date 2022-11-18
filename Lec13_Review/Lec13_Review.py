# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 16:50:56 2022

@author: rcuppari
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import glob

###############################################################################
###############################################################################
##                          Lecture 13: The End!                             ##
###############################################################################
###############################################################################

## REMINDERS: 
    ## please fill out the course evals! They help make this course better every year
    ## and hopefully you all realize I take feedback seriously 
    
    ## please submit your homeworks if you are missing any! There's no reason to get 
    ## an LP in this class
    
    ## since I am skipping gpd, would anyone show up if I held an extra class during 
    ## the reading days? Totally voluntary, and just if people are interested 
    ## ~ 2 hours
    


############################## DO FUN STUFF! ##################################
## using some data from: 
    ## Cholera data from WHO: https://worldhealthorg.shinyapps.io/page10cholera_data/
    ## Height data from Our World in Data: https://ourworldindata.org/human-height
    ## Food data from Our World in Data: https://ourworldindata.org/food-supply

cholera = pd.read_csv("Cholera_cases_2000_2021.csv")

## we can use "glob" to retrieve files -- we saw this a bit before 
## the asterisk means "anything" -- as in, find all files that start with 'avg' and then end 
## with '.csv', but they can have anything in the middle (we only have one here) 
filename = glob.glob("avg*.csv")
height = pd.read_csv(filename[0])

## printing columns is a good way to get a feeling for what is in a big dataset
print(cholera.columns)

## as is printing unique values -- here we are calling the column cause from our df
## and asking for all of the unique values within that column (our diseases)
print(cholera.WHO_Region.unique())

########################## IN-CLASS EXERCISE(S) ################################
## create a for loop that will make a plot with a line for each region's *summed* 
## *cholera* deaths compared to the year 

## first step: groupby region and year


## then plot so that each region has a different line 
## do this in a for loop, just for kicks! Use loc! 


## what about subsetting? And boxplots? 
countries = ['Argentina', 'Italy', 'Canada', 'United States', 'Uruguay']

## make a for loop which will retrieve these countries and *for each different 'year'*
## create a new barplot which shows the height in each country 
## so you should end up with 18 different figures 



## I also dug up a food supply (calories) dataset -- fun! 
## this could (should?) be related to height, so read in the data 
## (it's called cal'daily-per-cap-calories), pick a country,
## and then run a regression and plot a regression plot 
    ## reminder: sns.regplot() does the trick 
## suggestion: look at the countries that have the longest records 











