# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 21:33:06 2022

@author: rcuppari
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as sm

#############################################################################
####################### LECTURE 10: REGRESSIONS! ############################
#############################################################################

## GOALS: apply our Python skills to create statistical models

###############################################################################

## read in our dataset 
df = pd.read_csv('NC_SC_acid_rain_emissions.csv', sep=',', header=1)
df.columns = ['state', 'year', 'quarter', 'program', 'so2', 'nox', 'co2', 'heatinput']
df

## we can use plots to quickly check our data
sns.scatterplot('year', 'nox', data=df)
## what are we missing here for this to be a good plot?!

## we use ordinary least squares from the statsmodels package for our regression
## check out the notes for a very brief explainer on regressions 

### Define linear regression object, with equation written in quotes 
## no need to write constant term when we are using sm.ols 
lm_nox_year = sm.ols('nox ~ year', data=df)
lm_nox_year

### run regression by using .fit()
lm_nox_year_fit = lm_nox_year.fit()
lm_nox_year_fit

## to understand what our model looks like we can print the summary 
## with a bunch of useful stats/criteria that you should use to evaluate
## your model, including the adjusted r2, p-values, and measures of skew/kurtosis
## again, see lecture for some quick summaries
print(lm_nox_year_fit.summary())

### regression coefficients (betas)
params = lm_nox_year_fit.params
print(params)

## get predicted nox as a function of year based on regression
years = pd.DataFrame(np.arange(1995, 2021),columns=['year'])
prediction = lm_nox_year_fit.predict(years)
print(prediction.head())

## always visually check how your regression looks!
sns.scatterplot('year', 'nox', data=df)
plt.plot(years, prediction, color='k')
plt.ylabel('NOx emissions (tons)')
plt.xlabel('Year')

#################### IN-CLASS EXERCISE #####################
## you can also do this by defining a function and 
## calling the individual parameters -- build that function
def predict_nox_year():
    return 
## once you are done, use a different plot (hint: still a scatter)
## to show how well your regression works 

## create a second plot comparing the results of your first prediction
## and your prediction using the function you defined

############################################################
## now let's check if our regression pre-conditions are met... 

## start with normality 
plt.hist(df['nox'])

## what should we do if our data looks skewed like this? 

df['nox_log'] = np.log(df['nox'])
df.head()

## plot to see the difference
plt.figure()
sns.scatterplot('year', 'nox_log', data=df)
plt.ylabel('log(NOx emissions (tons))')
plt.xlabel('Year')

plt.figure()
plt.hist(df['nox_log'])

## we can also use a Q-Q test 
from statsmodels.graphics.gofplots import qqplot
fig = qqplot(df['nox_log'], line='s')

## or a shapiro-wilks test
## The null hypothesis here is that the data are normally distributed. 
## Given your appropriate p-value (using 0.01 here), 
## if the p-value is less than 0.01 
## then the data are **not** normally distributed. 
## If the p-value is greater, then they are. 

from scipy.stats import shapiro
stat, p = shapiro(df['nox_log'])
print('transformed data shapiro stats: ' + str(stat) + ', ' + str(p))
print()

stat_raw, p_raw = shapiro(df['nox'])
print('raw data shapiro stats: ' + str(stat_raw) + ', ' + str(p_raw))

## we can also use a box-cox transformation 
## This will transform the input data into a (more) normal distribution
## using a value scipy can calculate for us automatically: lambda. 
## Lambda is used in an equation which transforms the data 

from scipy.stats import boxcox
df['nox_bc'], lam = boxcox(df['nox'])
lam

plt.figure()
sns.scatterplot('year', 'nox_bc', data=df)
plt.ylabel('NOx emissions (transformed, unitless)')
plt.xlabel('Year')

plt.figure()
plt.hist(df['nox_bc'])

plt.figure()
fig = qqplot(df['nox_bc'], line='s')

## let's check our shapiro stat
stat, p = shapiro(df['nox_bc'])
print(stat, p)

## we should also check for constant variance 
resids = lm_nox_year_fit.resid
sns.scatterplot(df['year'], resids)

######################################################################
## when we are developing a regression, we should always use 
## different training and test sets 

## you can also use a different statsmodels package (sans "formula")
from sklearn.model_selection import train_test_split as tts
import statsmodels.api as sm
## tts will automatically split your X and Y variables into a consistent 
## training and test set
## NOTE: this will randomly split the data so time dependency is not 
## maintained -- that said, day of year or year can always be an input variable 


## tts takes your X inputs (can be multiple), your Y inputs, 
## the amount of the data you want as the test subset, 
## and then a "random state", which is to say it will randomly select 
## different subsets of the data for the training/testing 
X_train, X_test, y_train, y_test = tts(df['year'],df['nox_bc'], test_size = 0.2, random_state = 1)
print('Training X: ')
print(X_train.head())
print()
print('Training Y: ')
print(y_train.head())

## when you do use this version of sm, 
## it does not include a constant (intercept) variable, but we can easily add that 
df = sm.add_constant(df)
print(df.head())

X_train, X_test, y_train, y_test = tts(df[['year','const']],df['nox_bc'], test_size = 0.2, random_state = 1)
print('Training X: ')
print(X_train.head())
print()
print('Training Y: ')
print(y_train.head())

## develop your regression based on the training data
lm_nox_year = sm.OLS(y_train, X_train)
lm_nox_year_fit = lm_nox_year.fit()
print(lm_nox_year_fit.summary())

## now predict values using the test data
prediction = lm_nox_year_fit.predict(X_test)
predictX = lm_nox_year_fit.predict(X_train)
print(predictX.head())

## check how well your regression worked on the test values vs the training
plt.scatter(prediction, y_test,color="blue",label="Test Set")
plt.scatter(predictX, y_train,color="orange",label="Training Set")
#plt.plot(years, prediction, color='k')
plt.ylabel('NOx emissions (transformed, unitless)')
plt.xlabel('Predicted emissions')
plt.legend()

## let's go back to our original sm
import statsmodels.formula.api as sm
lm_nox_year = sm.ols('nox_bc ~ year', data=df)
lm_nox_year_fit = lm_nox_year.fit()
print(lm_nox_year_fit.summary())

prediction = predict_nox_year(lm_nox_year_fit.params, years)
sns.scatterplot('year', 'nox_bc', data=df)
plt.plot(years, prediction, color='k')
plt.ylabel('NOx emissions (transformed, unitless)')
plt.xlabel('Year')

## if we used the box-cox transformation, then we need to remember to 
## back transform our data once we have predicted it 
def box_cox_inverse(x, lam):
    if lam > 0:
        return (x * lam + 1) ** (1 / lam)
    elif lam == 0:
        return np.exp(x)
    else:
        print('Invalid lambda')
        
prediction_tons = box_cox_inverse(prediction, lam)
prediction_tons.head()

## check our back-transformed data
sns.scatterplot('year', 'nox', data=df)
plt.plot(years, prediction_tons, color='k')
plt.ylabel('NOx emissions (tons)')
plt.xlabel('Year')

###################### IN-CLASS EXERCISE ########################
## consider another input to our regression - the state 

## first print the different unique states (column state)

## next, create a scatter plot with year versus nox where the color
## of the markers is determined by the state. Is there a difference?

## now define a regression that includes year and state as predictors
## based on a **training set** of 60% of the data

## next, use your regression and predict the remaining 40%

## plot the predicted versus observed values

#################################################################

## some of you may be interested in interaction effects... 
lm_nox_year_state_int = sm.ols('nox_bc ~ year*state', data=df)
lm_nox_year_state_int_fit = lm_nox_year_state_int.fit()
print(lm_nox_year_state_int_fit.summary())

## are these significant? 
## let's try doing this with a function and calling the parameters
## note, if were were to continue using ols_fit.predict(), 
## we would need to put the inputs in a single dataframe

## and we want to define a binary "yes/no" variable for the state 
## because we only have two options 
years = np.arange(1995, 2021)
## since we just have two states, we can use a simple binary: SC will be 1 and NC will be 0 
SC_isSC = np.ones(len(years))
NC_isSC = np.zeros(len(years))

def predict_nox_year_state_int(params, years, SC_binary):
    return params[0] + params[1] * SC_binary + params[2] * years + params[3] * SC_binary * years

prediction_SC = predict_nox_year_state_int(lm_nox_year_state_int_fit.params, years, SC_isSC)
prediction_NC = predict_nox_year_state_int(lm_nox_year_state_int_fit.params, years, NC_isSC)

sns.scatterplot('year', 'nox_bc', data=df, hue='state')
plt.plot(years, prediction_SC, color='orange')
plt.plot(years, prediction_NC, color='blue')

###################### IN-CLASS EXERCISE ########################
## what if we want to check whether the quarter (season) is
## is a significant input? 

## well, first we want to consider it as a categorical versus
## integer variable (why??)

## so, start by using a list comprehension to make a new 
## column where the quarter is a *string* and not an integer 
## i.e., 0 --> "0", 1 --> "1", etc.. 

## now define a regression where year, state, and quarter are 
## all inputs 

#################################################################

## one reason we might have been interested in doing this 
## is because of our exploratory analysis... 
## let's make separate dfs for each state
df_nc = df.loc[df['state'] == 'NC', :]
df_sc = df.loc[df['state'] == 'SC', :]

plt.figure()
sns.scatterplot('year', 'nox_bc', data=df_nc, hue='quarter')
plt.figure()
sns.boxplot('quarter', 'nox_bc', data=df_nc)

## we can use an ANOVA test 
## the test returns the F statistic of the test along with the 
## p-value from the F distribution. A p-value greater than what 
## we set (say 0.05 for our purposes) means that the differences 
## in mean are not statistically significant, 
## while a p-value less than or equal to means that they are 
## statistically significant 

from scipy.stats import f_oneway
## nc versus sc 
f_oneway(df_nc['nox_bc'], df_sc['nox_bc'])

df_nc_q1 = df_nc.loc[df_nc['quarter'] == 1, :]
df_nc_q2 = df_nc.loc[df_nc['quarter'] == 2, :]
df_nc_q3 = df_nc.loc[df_nc['quarter'] == 3, :]
df_nc_q4 = df_nc.loc[df_nc['quarter'] == 4, :]

f_oneway(df_nc_q1['nox_bc'], df_nc_q2['nox_bc'], df_nc_q3['nox_bc'], df_nc_q4['nox_bc'])














