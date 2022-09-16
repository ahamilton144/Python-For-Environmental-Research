# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 08:17:14 2022

@author: rcuppari
"""
import pandas as pd 
import numpy as np 
import random

#############################################################################
################# LECTURE 5: NUMPY, PANDAS, AND REVIEW ######################
#############################################################################

## GOALS: explore everyone's favorite two packages 
## also, review for loops

###############################################################################
############################## FOR LOOPS!! ####################################


###############################################################################
################################ NUMPY!! ######################################
## one of the golden Python packages 
## it has a bunch of built in functions, but it's also useful because of 2D arrays
l = [[1, 2, 3, 4, 5], [10, 9, 8, 7, 6]]
print(l, type(l))
a = np.array(l)
print(a, type(a)) 

## NOTE: everything needs to be the same type
l2 = [[1, 2, '3'], [10, 9, 8]]
a2 = np.array(l2)
## what would you expect here?

## We can index just like a list! 
print(a2[1][0])
print(a2[1][1])

## or with a comma (note: still [rows, columns])
print(a2[1,0]) 

## or slices!
print(a2[:,2:])

## can find the dimensions 
print(a2.ndim)
print(a2.shape)

## it's easy to fill arrays with a single value 
threes = np.ones(3)
zeros = np.zeros((3, 10, 20))
print(zeros, zeros.shape)

## and it's got awesome sequence functions
print(np.arange(10))
print(np.arange(100, 223))
print(np.arange(10, -21, -2.5))

## and more.. 
print(np.random.normal(loc = 10, scale = 8, size = (10, 10)))
print()

## plus it's easier to do math than in lists, since it just 
## does the operation to each element 
a = np.array([1,2,3])
b = a + 3 
print(b) 
c = a*2
print(c)

################# IN CLASS EXERCISE! ###################
## make me a for loop that will iterate over every row 
## and every column, and print out the value of the array

################## LOGICAL INDEXING #####################
### create array of square roots, some of which are undefined
a = np.random.normal(loc = 10, scale = 8, size = (10, 10))
b = np.sqrt(a)
print(b)

### create boolean array with "True" everywhere there is an "nan" value, using np.isnan() function
c = np.isnan(b)
print(c)

### create a copy of b, then reset all nan's to 0 using logical indexing
d = b.copy()
d[c] = 0
# print(b)
print(d)

## how could we do the above with a for loop?

## all of this is similar to list comprehensions 
g = a > 3
print(g)

h = a[g]
print(h)

### often easier to put logical condition right inside brackets instead of defining as separate variable
g = e[e > 4]
print(g)

## we can also use multiple conditions
lt2 = a < 2 
gte4 = a >= 4
lt2_or_gte4 = lt2 | gte4 ## NOTE! We are using "|" NOT "or"

g = a[lt2_or_gte4]
print(g)
print(g.shape)

###############################################################################
############################### PANDAS!! ######################################
## truly my favorite package of them all, mostly because I love dataframes 
## to me, dfs (DataFrames) are the Excel spreadsheet of Python
## using the og well water example... 

## numeric index for each household in county
household = ['H' + str(i) for i in list(range(1000))]
print(household[:20])

## water source for each household
water = [random.choices(['municipal', 'private'], weights = [0.6, 0.4], k=1)[0] for h in household]
print(water[:20])

## income for each household 
income = [max(random.gauss(50000, 20000), 0) for h in household]
print(income[:20])

county = pd.DataFrame({'household': household, 'water': water, 'income': income})
county

## note: you can also set your index to different values
## and you can preview your df
county.head()
county.tail()
county.columns

## can specify a column 
county['water'].head()
county.water.head()

## and we can use a few different tools to add columns
## INCLUDING using numpy! 
nrow = county.shape[0]
print(nrow)
residents = np.random.choice([1, 2, 3, 4, 5, 6], size = nrow, p = [0.3, 0.25, 0.15, 0.15, 0.075, 0.075])
county['residents'] = residents
county.head()

## note: we could've used a for loop here but that would've been very inefficient

################# READING IN DATA!!!!! #################
## we use pandas! Either "pd.read_csv()" or "pd.read_excel()
## import data. "header=2" tells it to ignore the first two lines and use the 3rd line as the column names.
stocks = pd.read_excel('HistoricalStockPrices.xlsx', sheet_name = 'Combined', header = 2)
stocks.head()

stocks = pd.read_excel('HistoricalStockPrices.xlsx', sheet_name = 'Combined', header = 2, index_col = 0)
stocks.head()

## note the special index: 
print(stocks.index)

import matplotlib.pyplot as plt
plt.plot(stocks)
plt.legend(stocks.columns)
plt.xlabel('Year')
plt.ylabel('Price ($/share)')

################## IN-CLASS EXERCISE ####################
## calculate the relative growth of each stock
companies = []
for c in stocks.columns:
    ## want to append each column's name to the companies df
    ## DO THIS!
    ## also want to add a new column for growth
    
    ## WHAT'S THE INDEX??
    stocks[c + '_growth'] = ## this year / last year

print(companies)
print()
stocks.head()

stocks_stats = pd.DataFrame({'mean': stocks.mean(),
                             'std': stocks.std(),
                             'min': stocks.min(),
                             'max': stocks.max(),
                             'q05': stocks.quantile(0.05),
                             'q95': stocks.quantile(0.95)})
stocks_stats
## we can index in a few ways: 
## 1) make things a series (i.e., a pd structure with 1 column)
mean = stock_stats['mean']
print(type(mean))

## note, this has a name associated! So can just pull values
mean_vals = stocks_stats['mean'].values

## 2) we can use "iloc", which uses the integer position 
print(stocks_stats.iloc[3, 1])
stocks_stats.iloc[3, :] 

## 3) we can use "loc", which uses the label 
stocks_stats.loc['Amazon', 'std']
stocks_stats.loc[['Apple', 'Facebook', 'Facebook_Apple'], ['mean', 'min', 'max']]

## subsetting! We talked about this way back when 
is_private = county['water'] == 'private'
print(is_private)
print()

county_private = county.loc[is_private, :]
county_private

## any questions? Lots of questions? 








