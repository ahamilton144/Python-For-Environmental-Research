# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 22:19:56 2022

@author: rcuppari
"""

import pandas as pd
import os 


#################################################################
##  downloaded annual data from the American Community Survey  ##
##    specifically indicators S0801 and S0802, 2010 - 2021     ##
#################################################################

## let's read all files within our directory 
## and we are going to make three new dataframes: 
## male, female, and overall. I am using this for 

## use a for loop to iterate through all files in our directory without
## needing to type them all up 

## my function accepts a string which is meant to be a part of the name of the 
## series of csvs to read in -- something that all files for a specific dataset have
## while the prefix bit is just the name I want to use when I write out the file
## I am using it for two datasets -- one of which is *not* split by sex, 
## and so I need to specify whether or not to look for a male/female column
def read_in_directory(string, prefix, split_by_sex = True): 
    
    ## initialize dfs
    male_df = pd.DataFrame() ## unncessary if split_by_sex = False, so could put this in an if statement
    female_df = pd.DataFrame()
    total_df = pd.DataFrame()

    count = 0

    for file in os.listdir():
        # create if statement so that *if* the filename contains a specific string
        ## then it should be read in 
        if string in file: #file.endswith(".csv"):
            #print(file)
            file_path = f"{file}"
            ## I can look at the filenames and see that the year is there, 
            ## and automatically extract the location of the year integers 
            ## which is 7/8/9/10
            year = file_path[7:11]
            print(year)
            # call read text file function
            new_df = pd.read_csv(str(file))
            
            ## let's extract our columns of interest
            ## if I don't specify DataFrame, it reads in as a Series and messes up my 
            ## column names 
            new_tot = pd.DataFrame(new_df['North Carolina!!Total!!Estimate'])
            
            ## but also want to save our different groupings, without constantly 
            ## repeating them (assuming they are the same). So now we are saying that 
            # ## if it's our first read in, we save the grouping column 
            if count == 0: 
                labels = pd.DataFrame(new_df.iloc[:,0])
                total_df = pd.concat([labels, total_df], axis = 1)
                if split_by_sex == True: 
                    male_df = pd.concat([labels, male_df], axis = 1)
                    female_df = pd.concat([labels, female_df], axis = 1)
            count += 1  ## add to the counter so it does not happen in subsequent iterations
            
            
            ## change column names
            new_tot.columns = ['tot' + str(year)]
        
            ## and we can use concat here because we just want individual columns
            total_df = pd.concat([total_df, new_tot], axis = 1)
        
            ## now just doing all of the operations but for the male/female dfs
            ## **if** we have a dataset that is split by sex 
            if split_by_sex == True: 
                new_male = pd.DataFrame(new_df['North Carolina!!Male!!Estimate']) 
                new_female = pd.DataFrame(new_df['North Carolina!!Female!!Estimate'])

                new_male.columns = [str(year)] ## could make these 'male' + str(year) but I 
                ## decided we might want to do analysis that is easier with the same column names
                new_female.columns = [str(year)]
                
                male_df = pd.concat([male_df, new_male], axis = 1)
                female_df = pd.concat([female_df, new_female], axis = 1)
        
        
        if split_by_sex == True: ## only write out male and female if specified 
            male_df.to_csv(prefix + "_male.csv", encoding="cp1252")
            female_df.to_csv(prefix + "_female.csv", encoding="cp1252")        
        total_df.to_csv(prefix + "_overall.csv", encoding="cp1252")        
        
                        
        ## also, I did this once and got very weird values in the label columns
        ## and realized that it's because they have spaces in the original 
        ## Excel. So, I googled what could be happening (this has never happened
        ## to me) and it seems like an encoding issue. Aha!
        ## to see what the weirdness was, just remove the "encoding" bit so it 
        ## defaults to a different encoding

        
## don't need to return anything because within the function we are writing a csv output        
read_in_directory('S0802', 'demo_commuters', split_by_sex = False)
read_in_directory('S0801', 'means_transport', split_by_sex = True)