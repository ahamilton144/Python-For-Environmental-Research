# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 17:49:26 2022

@author: rcuppari
"""
 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols

#############################################################################
################## LECTURE 12: TIME SERIES ANALYSIS CONT ####################
#############################################################################

## GOALS: wrap up Lecture 11
## QUESTION: do y'all want a short refresher recording for 
## last week + this week's lectures?
    
###############################################################################

## read today's data
sacra_weather = pd.read_csv("sacramento_wind_temp.csv")
print(sacra_weather.head())
sacra_weather['date'] = pd.to_datetime(sacra_weather.iloc[:,0])

shasta = pd.read_csv('shasta_inflow.csv', header=2)
print(shasta.head())

## we just care about inflow column
shasta = shasta.iloc[:, 0:2]
shasta.columns = ['date', 'inflow']
shasta.head()

## transform date to datetime type and set index 
##   (see here for datetime format specifiers: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior)
shasta['date'] = pd.to_datetime(shasta['date'], format='%b-%y')
## rename inflow column 
shasta.index = shasta['date']
shasta = shasta.iloc[:, [1]]
shasta.head()

## let's make our column numeric 
shasta['inflow_new'] = 0 
for i in range(0,len(shasta['inflow'])): 
    new = shasta['inflow'].iloc[i]
    shasta['inflow_new'].iloc[i] = new.replace(',','')
print(shasta.head())
print()

## now convert to numeric type
shasta['inflow_new'] = pd.to_numeric(shasta['inflow_new'])

###################### TIMESERIES ANALYSIS ##################################
## now for our timeseries analysis: 

## create time step variable (note order of dates)
shasta['time'] = list(range(shasta.shape[0]))
shasta.iloc[:20,:]


### fit to lognormal distribution
from scipy.stats import lognorm, gamma, norm

ln_shape, ln_loc, ln_scale = lognorm.fit(shasta['inflow_new'])
ln_shape, ln_loc, ln_scale

### check fit
shasta['inflow_new'].mean()
print(f"Mean: data = {shasta['inflow_new'].mean()}, fit = {lognorm.mean(ln_shape, ln_loc, ln_scale)}")
print(f"Std: data = {shasta['inflow_new'].std()}, fit = {lognorm.std(ln_shape, ln_loc, ln_scale)}")
      
print('Std: ' + str(shasta['inflow_new'].std()))

### plot theoretical distribution vs histogram
flow = np.arange(0, 400000, 1000)
ln_pdf = lognorm.pdf(flow, ln_shape, ln_loc, ln_scale)

plt.hist(shasta['inflow_new'], density=True, bins=30, color = 'orange')
plt.plot(flow, ln_pdf, color = 'black')

### now generate a synthetic record of length 480 months (40 years)
shasta['synthetic1_inflow'] = lognorm.rvs(ln_shape, ln_loc, ln_scale, size=480)
shasta['synthetic1_inflow']

############################################################
### what about seasonality? we know that flow varies in predictable way from season to season.
shasta['month'] = shasta.index.month

monthly_mean = shasta.groupby('month').mean()['inflow_new']
monthly_std = shasta.groupby('month').std()['inflow_new']
monthly_std

monthly_mean_synthetic1 = shasta.groupby('month').mean()['synthetic1_inflow']
monthly_std_synthetic1 = shasta.groupby('month').std()['synthetic1_inflow']

plt.plot(monthly_mean, color='k')
plt.plot(monthly_mean + monthly_std, color='k', ls=':')
plt.plot(monthly_mean - monthly_std, color='k', ls=':')

plt.plot(monthly_mean_synthetic1, color='orange')
plt.plot(monthly_mean_synthetic1 + monthly_std_synthetic1, color='orange', ls=':')
plt.plot(monthly_mean_synthetic1 - monthly_std_synthetic1, color='orange', ls=':')

plt.xlabel('month')
plt.ylabel('inflow (AF)')


############################################################
### Want to account for seasonality and autocorrelation. First log transform.
shasta['log'] = np.log(shasta['inflow_new'])
plt.plot(shasta['log'])

import statsmodels.formula.api as sm
### There are a few ways to deal with any trends, but we will "deseasonalize" data 
## by converting to monthly z-scores
monthly_mean_log = shasta.groupby('month').mean()['log']
monthly_std_log = shasta.groupby('month').std()['log']

shasta['deseas'] = shasta['log'].copy()
for i in range(1, 13):
    mu = monthly_mean_log[i]
    sigma = monthly_std_log[i]
    shasta['deseas'].loc[shasta['month'] == i] = (shasta['deseas'].loc[shasta['month'] == i] - mu) / sigma
    
plt.figure()
plt.plot(shasta['deseas'])

plt.figure()
plt.hist(shasta['deseas'])

### This looks more random! But let's check whether autocorrelation seems significant
from statsmodels.graphics.tsaplots import plot_acf
plt.figure()
fig = plot_acf(shasta['deseas'])

### create lag variables to deal with auto-correlation
shasta['deseas_l1'] = np.nan
shasta['deseas_l2'] = np.nan
shasta['deseas_l3'] = np.nan
shasta['deseas_l12'] = np.nan
print(shasta.head())
print()

shasta['deseas_l1'].iloc[1:] = shasta['deseas'].values[:-1]
shasta['deseas_l2'].iloc[2:] = shasta['deseas'].values[:-2]
shasta['deseas_l3'].iloc[3:] = shasta['deseas'].values[:-3]
shasta['deseas_l12'].iloc[12:] = shasta['deseas'].values[:-12]

shasta.head

### set up autoregressive models with different lags
lm_log_ar = sm.ols('deseas ~ deseas_l1 + deseas_l2 + deseas_l3 + deseas_l12', data=shasta)
lm_log_ar_fit = lm_log_ar.fit()
print(lm_log_ar_fit.summary())

### set up autoregressive models with different lags
lm_log_ar = sm.ols('deseas ~ deseas_l1 + deseas_l2 + deseas_l3 + deseas_l12', data=shasta)
lm_log_ar_fit = lm_log_ar.fit()
print(lm_log_ar_fit.summary())

### set up autoregressive models with different lags
lm_log_ar = sm.ols('deseas ~ deseas_l1 + deseas_l2', data=shasta)
lm_log_ar_fit = lm_log_ar.fit()
print(lm_log_ar_fit.summary())

### set up autoregressive models with different lags
lm_log_ar = sm.ols('deseas ~ deseas_l1', data=shasta)
lm_log_ar_fit = lm_log_ar.fit()
print(lm_log_ar_fit.summary())

### set up autoregressive models with different lags
lm_log_ar = sm.ols('deseas ~ deseas_l1 + deseas_l12', data=shasta)
lm_log_ar_fit = lm_log_ar.fit()
print(lm_log_ar_fit.summary())
### although deseas_l12 is not significant, this one is still best in terms of AIC/BIC, so let's go with it.

### now look at residuals to see if they look like "white noise" (uncorrelated & normally distributed)
shasta['ar_resid'] = lm_log_ar_fit.resid
print(shasta)

plt.figure()
plt.plot(shasta['ar_resid'])

plt.figure()
plt.hist(shasta['ar_resid'])

plt.figure()
fig = plot_acf(shasta['ar_resid'].iloc[12:])

### ok, so we have "whitened" data! Now what? Now we can turn around and create synthetic data by reversing the process

## first, generate synthetic normally distributed data:
shasta['synthetic2_noise'] = norm.rvs(shasta['ar_resid'].mean(), shasta['ar_resid'].std(), size=shasta.shape[0])

plt.plot(shasta['ar_resid'])
plt.plot(shasta['synthetic2_noise'])

### now apply the AR relationship. here is a function to get regression prediction
def predict_ar(lag1, lag12):
    return lm_log_ar_fit.params[0] + lm_log_ar_fit.params[1] * lag1 + lm_log_ar_fit.params[2] * lag12

### We need 12 years of lagged data to start, we can use the last 12 years of real data
max_lag = 12
nrow = shasta.shape[0]

synth_ar = list(shasta['deseas'].iloc[-max_lag:])
synth_ar

### now add prediction plus noise to get AR time series
for i in range(nrow):
    lag12 = synth_ar[i]
    lag1 = synth_ar[i + max_lag - 1]
    prediction = predict_ar(lag1, lag12)
    noise = shasta['synthetic2_noise'].iloc[i]
    synth_ar.append(prediction + noise)
    
shasta['synthetic2_deseas'] = synth_ar[-nrow:]
shasta

### this synthetic time series (noise + AR relationship) looks similar to our original deseasonalized data
plt.plot(shasta['deseas'])
plt.plot(shasta['synthetic2_deseas'])

### next step is to reseasonalize by converting from monthly z-scores back to log-scale data
shasta['synthetic2_log'] = shasta['synthetic2_deseas'].copy()
for i in range(1, 13):
    mu = monthly_mean_log[i]
    sigma = monthly_std_log[i]
    shasta['synthetic2_log'].loc[shasta['month'] == i] = shasta['synthetic2_log'].loc[shasta['month'] == i] * sigma + mu
    
plt.plot(shasta['log'])
plt.plot(shasta['synthetic2_log'])

### lastly, we take the exp to convert from log scale back to inflows in AF
shasta['synthetic2_inflow'] = np.exp(shasta['synthetic2_log'])

### plot original data vs synthetic1 vs synthetic2
plt.figure()
plt.plot(shasta['inflow_new'], color='k', label='historic')
plt.plot(shasta['synthetic1_inflow'], color='royalblue', label='synth1')
plt.plot(shasta['synthetic2_inflow'], color='firebrick', label='synth2')
plt.legend()

plt.figure()
plt.plot(shasta['inflow_new'].iloc[60:120], color='k', label='historic')
plt.plot(shasta['synthetic1_inflow'].iloc[60:120], color='royalblue', label='synth1')
plt.plot(shasta['synthetic2_inflow'].iloc[60:120], color='firebrick', label='synth2')
plt.legend()

### look at monthly statistics (log space)
monthly_mean_synthetic2 = shasta.groupby('month').mean()['synthetic2_inflow']
monthly_std_synthetic2 = shasta.groupby('month').std()['synthetic2_inflow']

plt.plot(monthly_mean, color='k', label='historic')
plt.plot(monthly_mean + monthly_std, color='k', ls='--')
plt.plot(monthly_mean - monthly_std, color='k', ls=':')

plt.plot(monthly_mean_synthetic1, color='royalblue', label='synth1')
plt.plot(monthly_mean_synthetic1 + monthly_std_synthetic1, color='royalblue', ls='--')
plt.plot(monthly_mean_synthetic1 - monthly_std_synthetic1, color='royalblue', ls=':')

plt.plot(monthly_mean_synthetic2, color='firebrick', label='synth2')
plt.plot(monthly_mean_synthetic2 + monthly_std_synthetic2, color='firebrick', ls='--')
plt.plot(monthly_mean_synthetic2 - monthly_std_synthetic2, color='firebrick', ls=':')

plt.xlabel('month')
plt.ylabel('inflow (AF)')
plt.legend()

############ HOW CAN WE CHECK THAT THESE ARE/ARE NOT DIFFERENT? ###############
## well, we already did, but let's use this as an opportunity to run through 
## how to do a one-way ANOVA test
## null hypothesis: these two populations are equal 
from scipy.stats import f_oneway
import statsmodels.api as sm

f_oneway(shasta['inflow_new'], shasta['synthetic1_inflow'], shasta['synthetic2_inflow'])
## returns the f and p values

## more in depth... 
lm = ols("inflow_new ~ synthetic1_inflow + synthetic2_inflow", shasta).fit() ## notice that I am doing the fit in one step
sm.stats.anova_lm(lm)


################################################################################

### Now what can we do with this? Run it through (very simplified) Shasta Reservoir!
storage_max = 4.5e6    ## max storage at shasta is ~4.5 million AF
storage_tmin1 = 2.25e6   ## let's assume it starts half full
inflow = list(shasta['inflow_new'])   ## inflow in AF/month
demand = shasta['inflow_new'].mean() * 0.8  ## assume demand is 80% of average inflow

### function for releases based on demand and max storage capacity
def get_release_storage(storage_tmin1, storage_max, inflow, demand):
    release = demand
    storage_t = storage_tmin1 + inflow - release
    if storage_t < 0:
        release += storage_t
        storage_t = 0
    elif storage_t > storage_max:
        release += (storage_t - storage_max)
        storage_t = storage_max
    return release, storage_t

release_1, storage_1 = get_release_storage(storage_tmin1, storage_max, inflow[0], demand)
release_1, storage_1

### run through simulation & store results in pandas
shasta['release'] = -1.
shasta['storage'] = -1.

for t in range(nrow):
    release, storage_t = get_release_storage(storage_tmin1, storage_max, inflow[t], demand)    
    shasta['release'].iloc[t] = release
    shasta['storage'].iloc[t] = storage_t
    storage_tmin1 = storage_t

shasta

plt.figure()
plt.plot(shasta['release'])

plt.figure()
plt.plot(shasta['storage'])

### repeat for synthetic data
storage_max = 4.5e6    ## max storage at shasta is ~4.5 million AF
storage_tmin1 = 2.25e6   ## let's assume it starts half full
inflow = list(shasta['synthetic2_inflow'])   ## inflow in AF/month
demand = shasta['inflow_new'].mean() * 0.8  ## assume demand is 80% of average (historical) inflow

shasta['synthetic2_release'] = -1.
shasta['synthetic2_storage'] = -1.

for t in range(nrow):
    release, storage_t = get_release_storage(storage_tmin1, storage_max, inflow[t], demand)    
    shasta['synthetic2_release'].iloc[t] = release
    shasta['synthetic2_storage'].iloc[t] = storage_t
    storage_tmin1 = storage_t

plt.figure()
plt.plot(shasta['synthetic2_release'])

plt.figure()
plt.plot(shasta['synthetic2_storage'])

### what if demand is expected to grow by 0.1%/month?
### repeat for synthetic data
storage_max = 4.5e6    ## max storage at shasta is ~4.5 million AF
storage_tmin1 = 2.25e6   ## let's assume it starts half full
inflow = list(shasta['synthetic2_inflow'])   ## inflow in AF/month

demand_0 = shasta['inflow_new'].mean() * 0.8  ## assume demand is 80% of average (historical) inflow at start
demand = [demand_0 * (1.001)**t for t in range(nrow)]
# demand

shasta['synthetic2_release'] = -1.
shasta['synthetic2_storage'] = -1.

for t in range(nrow):
    release, storage_t = get_release_storage(storage_tmin1, storage_max, inflow[t], demand[t])    
    shasta['synthetic2_release'].iloc[t] = release
    shasta['synthetic2_storage'].iloc[t] = storage_t
    storage_tmin1 = storage_t

plt.figure()
plt.plot(shasta['synthetic2_release'])

plt.figure()
plt.plot(shasta['synthetic2_storage'])

## Let's generate 10,000 different synthetic inflow scenarios, each of which is 20 years (240 months) long. 
## We will store the final synthetic inflows in a 10,000x240 NumPy array. We want to use NumPy arrays rather 
## than Pandas for all synthetic generation steps (e.g. white noise, deseasonalized residual, etc), since this 
## is significantly faster and simpler when dealing with a lot of repitition.

## variables for synthetic data
nyear = 20
nmonth = nyear * 12
nsim = 10000

## first, generate synthetic normally distributed data:
## BABY IN CLASS EXERCISE! Who can dissect this statement?
noise = norm.rvs(shasta['ar_resid'].mean(), shasta['ar_resid'].std(), size=(nsim, nmonth))

print(noise.shape)
print(noise)

### now apply the AR relationship. here is a function to apply regression prediction
def predict_ar(lag1, lag12):
    return lm_log_ar_fit.params[0] + lm_log_ar_fit.params[1] * lag1 + lm_log_ar_fit.params[2] * lag12

### We need 12 years of lagged data to start, we can use the last 12 years of real data
max_lag = 12
deseas = np.zeros((nsim, nmonth+max_lag))
deseas[:, :max_lag] = list(shasta['ar_resid'].iloc[-max_lag:])
### now add prediction plus noise to get AR time series
for i in range(nmonth):
    lag12 = deseas[:, i]
    lag1 = deseas[:, i + max_lag - 1]
    prediction = predict_ar(lag1, lag12)
    deseas[:, i + max_lag] = prediction + noise[:, i]

deseas = deseas[:, max_lag:]
    
print(deseas.shape)
print(deseas)

## now reseasonalize using the monthly means and stds, to get to log-scale inflows
month = [t % 12 + 1 for t in range(nmonth)]
log = deseas.copy()
for i in range(nmonth):
    mu = monthly_mean_log[month[i]]
    sigma = monthly_std_log[month[i]]
    log[:, i] = log[:, i] * sigma + mu
print(log.shape)
print(log)

### lastly, exponentiate to get back to original scale (AF/month)
synth_inflow = np.exp(log)

print(synth_inflow.shape)
print(synth_inflow)

### plot next to original data to make sure it looks ok
for s in range(10):
    plt.plot(synth_inflow[s, :], alpha=0.3)
plt.plot(shasta['inflow_new'].iloc[:nmonth].values, color='k', lw=2)

### save data 
np.savetxt('synth_inflow.csv', synth_inflow, delimiter=',')










