# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 15:17:36 2022

@author: rcuppari
"""

import numpy as np
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns

#############################################################################
######################## LECTURE 8: PLOTTING!!! #############################
#############################################################################

## GOALS: reinforce how we manipulate data and find nice ways to visualize 
## said data 

###############################################################################

## we can set standard fonts/other parameters
## adjust default plot font size
matplotlib.rcParams.update({'font.size': 18})

## let's start with random data that we made up for a parabola
def parabola(a, b, c, x):
    return a + b * x + c * x **2

X = np.arange(-10, 10)
print(X)

Y1 = [parabola(-2, 0, 1, x) for x in X]
Y2 = [parabola(5, -2, -0.5, x) for x in X]
print(Y2)

### simple line plot -- plt.plot(X_VALUES, Y_VALUES)
plt.plot(X, Y1)

## we can overlay however many lines we want on a single plot
### plot Y1 & Y2 together, and fix figure size
fig = plt.figure(figsize = (10, 6))
plt.plot(X, Y1)
plt.plot(X, Y2)

## we can also create subplots so that we create multiple separate plots
### plot Y1 & Y2 separately from 1 cell
fig = plt.figure(figsize = (10, 6))
plt.plot(X, Y1)
fig = plt.figure(figsize = (10, 6))
plt.plot(X, Y2)

## let's make our figure nicer though -- we can add labels with different font sizes
fig = plt.figure(figsize = (10, 6))
plt.plot(X, Y1, label='First parabola')
plt.plot(X, Y2, label='Second parabola')
plt.xlabel('X-axis',fontsize=40)
plt.ylabel('Y-axis',fontsize=40)
plt.legend()

## or also change linewidth/style/color -- you'll need to investigate matplotlib.pyplot
## for the full set of customizable bits
fig = plt.figure(figsize = (10, 6))
plt.plot(X, Y1, label='First parabola', color='mediumturquoise', linestyle='--', linewidth=4, alpha=0.9)
plt.plot(X, Y2, label='Second parabola', color='mediumslateblue', marker='d', alpha=0.4)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(loc='upper center')

########################### IN-CLASS EXERCISE #################################

#Let's create some fake data to represent annual snowfall in two locations, one colder and one warmer.

# 1. Create an x variable that includes the integers from 1920 to 2020.
# 2. Create one y variable that is random draws from an exponential distribution with a scale parameter of 10, with the same length as x. (this is the colder climate)
# 3. Create a second y variable that is random draws from a lognormal distribution with a mean of 0.5 and a standard deviation of 1, with the same length as x. (this is the warmer climate)
# 4. Plot both time series. Label the x axis "Year" and the y axis "Annual snowfall (inches)". Create a legend where the first series is labeled "Colder" and the second is labeled "Warmer". Color them using colder and warmer-looking colors of your choice.

############################## REAL DATA! #####################################
### read in data. ignore first row ("header") where I have written info about where data was downloaded.
df = pd.read_csv('chapel_hill_weather.csv', header = 1)
df.head()

### Organize data
## rename columns
df.columns = ['station', 'name', 'date', 'precip', 'snow', 'tmax', 'tmin']
df.head()

## convert date column to datetime type -- this lets us automatically sort the date
## and has python recognize months versus years versus minutes etc. 
df.date = pd.to_datetime(df['date'])
df.head()

## it can be helpful to index by date so that can be your x axis 
df.index = df['date']
# remove unnecessary columns
df = df.loc[:, ['precip', 'snow', 'tmax', 'tmin']]
df.head()

## Get year, month, day for each
df['year'] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day
df

## Only keep data since 1970
df = df.loc[df['year'] >= 1970, :]
df

### Clean data -- most of your data is going to be dirty in real life 
# remove NANs by assuming previous value (note, this is not the most sophisticated way to fill data)
nrows = df.shape[0]
ncols = df.shape[1]
np.isnan(df)

print("Fraction of nan's before:")
print(np.isnan(df).sum(axis=0) / nrows)
print()


### replace nan's with previous value in time
for i in range(nrows):
    for j in range(ncols):
        if np.isnan(df.iloc[i, j]):
            df.iloc[i, j] = df.iloc[i - 1, j]
            
print("Fraction of nan's after:")
print(np.isnan(df).sum(axis=0) / nrows)


### Plot data as time series
plt.plot(df.precip)

plt.figure()
plt.plot(df.snow)

plt.figure()
plt.plot(df.tmax)

plt.figure()
plt.plot(df.tmin)

## Looks like a bad data value - tmin probably wasn't >120 in January
df.loc[df.tmin > 100, :]

## Let's look at that data in context of neighbors
i = np.argmax(df.tmin)
print(i)
df.iloc[(i - 2):(i + 3), :]
## if you get an error here it may be because of which version of Python you are using. In that case, try using 
## pd.Timedelta()

## Let's reset that value based on previous value (note: you could also do average, or leave blank)
df.iloc[i, 3] = df.iloc[i - 1, 3]
df.iloc[(i - 2):(i + 3), :]
### Note: this particular warning about "setting a value based on a copy of a slice" occurs a lot, personally I ignore it

plt.plot(df.tmin)


#################### OTHER COOL WAYS TO VISUALIZE DATA!!! #####################
## Function for plotting time series over period
## note -- no return here, we are just making the figure and it will pop up 
def plot_weather(df, colname, ylabel, startyear=df.year.min(), endyear=df.year.max()):
    data = df.loc[(df['year'] >= startyear) & (df['year'] <= endyear), colname]
    fig = plt.figure(figsize = (12,8))
    plt.plot(data)
    plt.xlabel('Year')
    plt.ylabel(ylabel)
    
plot_weather(df, 'precip', 'Precip (mm)')

## Function for plotting multiple time series over period
colors=['blue','orange','steelblue','red']
def plot_weather_multi(df, colnames, ylabel, color, startyear=df.year.min(), endyear=df.year.max()):
    data = df.loc[(df['year'] >= startyear) & (df['year'] <= endyear)]
    fig = plt.figure(figsize = (12,8))
    for i in range(len(colnames)):
    ## as I was mentioning in class, we want to change this for loop to use the index, 
    ## NOT the name of the column so that we can change  the colors as we go
    ## to do so, we need to go through the length of columns, NOT the actual names (which is what i in colnames will give)
    ## and to do so we actually need to put our interested columns into a series/df called something
        column = colnames[i]
        plt.plot(data.loc[:,column], label=colnames[i], color = colors[i], alpha=0.8)
    plt.xlabel('Year')
    plt.ylabel(ylabel)
    plt.legend()
    
names = ['tmin', 'tmax', 'precip']
plot_weather_multi(df, names, 'Values',colors, 1998, 2002)

########################### IN-CLASS EXERCISE #################################
# Make a line plot with both snow and precipitation over the entire timeseries 
# without the defined function, making sure to include axis labels. 
# Make the color of the snow line blue and the color of the rain line green.

    
########################### MORE FUN PLOTS ###################################
## How do min and max temperatures compare? Scatterplot.
fig = plt.figure(figsize = (12,8))
plt.scatter(df['tmin'], df['tmax'])
plt.xlabel('tmin')
plt.ylabel('tmax')

## customizing with color and transparency
fig = plt.figure(figsize = (12,8))
plt.scatter(df['tmin'], df['tmax'], color='k', alpha=0.1)
plt.xlabel('tmin')
plt.ylabel('tmax')

## what about scattering by month?
markers = ['o', '<', '^', '>', '<', 's', 'p', 'P', '*', 'h', 'X', 'D']
colors = ['navy', 'mediumslateblue', 'turquoise', 'springgreen', 'forestgreen', 'greenyellow', 'yellow', 'gold', 'orange', 'orangered', 'firebrick', 'mediumvioletred']
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

fig = plt.figure(figsize = (12,8))
for i in range(12):
    data = df.loc[df['month'] == (i+1), :]
    plt.scatter(data['tmin'], data['tmax'], color=colors[i], marker=markers[i], alpha=0.3, label=months[i])
plt.xlabel('tmin')
plt.ylabel('tmax')
plt.legend()

## histogram of tmin vs tmax
fig = plt.figure(figsize = (12,8))
plt.hist(df['tmin'], color='b', alpha=0.5, label='tmin')
plt.hist(df['tmax'], color='r', alpha=0.5, label='tmax')
plt.xlabel('Temperature (F)')
plt.ylabel('Count')
plt.legend()

##### seaborn is another plotting package with some nice clean functions. plays nice with matplotlib parameters.
## kde plot
fig = plt.figure(figsize = (12,8))
for k in seasons:
    is_season = [m in season_months[k] for m in df['month']]
    data = df.loc[is_season, :]
    sns.kdeplot(data['tmax'], color=colors[k], shade=True, label=k)
plt.xlabel('Max temperature (F)')
plt.ylabel('Density')
plt.legend()

## boxplots
sns.boxplot(x = df['month'], y = df['tmax'])
plt.legend

########################### IN-CLASS EXERCISE #################################
# Make a figure with two subplots: one with a density plot and one with a histogram
# for snowfal 

##################### IF WE HAVE TIME... Q&A ON PANDAS ########################







