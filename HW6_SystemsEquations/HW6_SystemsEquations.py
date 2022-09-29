# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 11:20:49 2022

@author: rcuppari
"""

## MAKE SURE TO SAVE THIS WITH YOUR NAME AND PREFERABLY CHANGE THE 
## AUTHORSHIP AT THE TOP (i.e., where it says rcuppari, put your onyen!)
## let's import everything we need at the top
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import Bounds, LinearConstraint, minimize

## here is the set up for the problem (check out the Jupyter Notebook for 
## more context)
demand = (1000, -20)
coal = (20, 0.2, 0)
gas = (5, 1, 0)
renew = (0, 2., 0)

def mb_demand(q):
    return demand[0] + demand[1] * q

def mc_coal(q):
    return coal[0]  + coal[1] * q

def mc_gas(q):
    return gas[0] + gas[1] * q

def mc_renew(q):
    return renew[0] + renew[1] * q

q_grid = np.arange(0, 200)
plt.plot(q_grid, [mb_demand(q) for q in q_grid])
plt.plot(q_grid, [mc_coal(q) for q in q_grid])
plt.plot(q_grid, [mc_gas(q) for q in q_grid])
plt.plot(q_grid, [mc_renew(q) for q in q_grid])
plt.legend(['MB Demand', 'MC Coal', 'MC Gas', 'MC Renewable'])
plt.xlabel('Quantity (MWh)')
plt.ylabel('Price ($/MWh)')
plt.xlim([0, 60])
plt.ylim([0, 120])

################################## Q1 ####################################
## 1) Use the Numpy linalg solution method from class to find the 4 quantity
## 𝑞's and the price of power 𝑝.




################################## Q2 ####################################
## 2) Use the minimize function from scipy.optimize to find the qs that 
## maximize net benefits, and the resulting price. 
## You should get the same answers (within rounding error) as problem 1
## Note: you don't need to use the quad function for optimization, 
## since we integrated the MNB by hand above, but you can if you prefer

## Hint: this is how you write the equality constraint, q_d - q_c - q_g - q_r (+0*p) = 0.
## You will need to add the second equality constraint to this function
linear_constraint = LinearConstraint([1, -1, -1, -1, 0], 0, 0)





################################## Q3 ####################################
## With Q3 (and Q4) we are going to look at how a carbon tax versus cap 
## could influence our model outputs 
## For Q3, add a carbon tax that results in an increase of $10/MWh for 
## coal-powered electricity and $5/MWh for natural gas-powered electricity. 
## This means you will have to change the marginal net benefit functions 
## for the fossil fuel suppliers:
## MNBc = p - 30 - 0.2*qc
## MNBg = p - 10 - qg





################################## Q4 ####################################
## Add a carbon cap of 10,000 lb CO2 for this system, 
## assuming that coal-fired electricity produces 735 lb/MWh and 
## gas-fired electricity produces 400 lb/MWh. In words, this just means that you
## are adding a new linear constraint: 735*qc + 400*qg <= 10,000





################################## Q5 ####################################
## how does the carbon tax versus the carbon cap impact electricity produced
## by each source? How does it affect total electricity demand and the price
## of electricity? 




