# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 09:58:02 2022

@author: rcuppari
"""

#############################################################################
#################### LECTURE 7: SYSTEMS OF EQUATIONS ########################
#############################################################################

## GOALS: learning to optimize systems of equations!
## also, for anyone taking 755, give you the skills to do the homework

###############################################################################

## a non script interlude: what is a system of equations 
## this is not a math class, so this will be an extremely short introduction 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 14})

### set up params for each sector, (constant, slope).
city = (150, -3)
farm = (70, -1)
indus = (80, -2)

### function for MNB, based on parameter tuple and quantity of water
def get_mnb(params, q):
    return params[0] + params[1] * q
get_mnb(city, 40)

### get MNB curve for each sector, over a grid of q points
qs = np.arange(0, 81)
city_q = [get_mnb(city, q) for q in qs]
farm_q = [get_mnb(farm, q) for q in qs]
indus_q = [get_mnb(indus, q) for q in qs]
print(indus_q)

### plot MNB curves
plt.plot(qs, city_q)
plt.plot(qs, farm_q)
plt.plot(qs, indus_q)
plt.xlim([0, 80])
plt.ylim([0, 150])
plt.xlabel('Quantity of water (acre-feet)')
plt.ylabel('Marginal net benefit (\$/acre-foot)')
plt.legend(['City','Farm','Industrial'])

## that was for UNLIMITED water -- everyone uses water up until the 
## the marginal cost of that water is equal to the marginal benefit 
## put in other words, they will use the water until they are paying 
## more for it than they are receiving. 
## another way to put it is to say that the marginal NET benefit is 
## equal to zero (net benefit = benefit - cost)

## but what if we have LIMITED quantities of water and there are many 
## people/sectors who want to use it? In that case some (or all) actors
## may get less water than they wanted. We can decide those allocations 
## by setting all sectoral marginal net benefits equal to each other 
## essentially we are saying that every unit of water goes to the most 
## beneficial use 

### solve simple lin sys eqns with numpy
## we use a constraint now -- qc + qf + qi <= 100 (writing this out on board)
A = np.array([[3, -1, 0], [0, -1, 2], [1, 1, 1]])
print(A)

b = np.array([80, 10, 100])
print(b)

x = np.linalg.solve(A, b)
x
print(x.sum())
print(np.sum(x))

## our MNBs should be the same for all sectors, even though the water 
## allocations will not be 
print( f'City MNB: ${round(get_mnb(city, x[0]), 1)}/acre-foot')
print( f'City water allocation: {round(x[0],1)} acre-feet of water')
print()
print( f'Farm MNB: ${round(get_mnb(farm, x[1]), 1)}/acre-foot')
print( f'Farm water allocation: {round(x[1],1)} acre-feet of water')
print()
print( f'Industrial MNB: ${round(get_mnb(indus, x[2]), 1)}/acre-foot')
print( f'Industrial water allocation: {round(x[2],1)} acre-feet of water')

## in many cases, we need to solve non-linear equations (i.e., with powers)
## for those of you taking 755, you are using the Cobb-Douglas Power function
## we can still solve our systems of equations, but we need a new tactic 
## we still need 1 equation per unknown, and now let's play around with 
## 5 water users. Our unknowns are the water allocations for each sector
## let's limit total water usage to 2000 acre-feet (and make this binding)

### import "root" function, which finds the roots of multivariate functions
from scipy.optimize import root 

### Define our system of equations as a function. Input q is 5-dimensional array of allocations, [q_a, q_b, q_c, q_d, q_e].
### The output, eq, is a 5-dimensional array that should be [0., 0., 0., 0., 0.] if q is a root.
def soe(q):
    eq = np.zeros(5)
    eq[0] = -85 * q[0] + 50 * q[1] - 4500
    eq[1] = -50 * q[1] + 6 * q[2] ** 2 + 2500
    eq[2] = -6 * q[2] ** 2 + 150 * q[3] ** (1/8) + 4000
    eq[3] = -150 * q[3] ** (1/8) + 40 * q[4] - 2000
    eq[4] = q.sum() - 2000
    return eq

### Supply initial guess along with our soe function, and then find root
guess = np.array([402., 401., 400., 399., 398.])
sol = root(soe, guess)

print(sol)
print()

q_allocation = sol['x']
print(q_allocation)

q_allocation.sum() ## our total allocations are 2000 which means we did this right!


## now let's think about a situation in which we have a constraint on total 
## water allocations AND a constraint on just a specific sector 
## say qd <= 1500
## so this time, let's not making our constraint binding (==). Let's actually 
## use the <= 
## to do this, we have to switch our strategy to MAXIMIZING net benefits 
## instead of setting all of the marginal net benefits equal 
## that said, we will be minimizing the negative version of our net benefits
## since the scipy.optimize function works better with mins
## write function for minus total net benefits. We make it negative since optimizer likes to minimize, not maximize.

## we are - through our code - integrating this function to find the net benefits
## (area under the net benefit curve)
def MinusNetBenefits(q):
    nb = np.zeros(5)
    nb[0] = 3000 * q[0] - 85/2 * q[0] ** 2
    nb[1] = 7500 * q[1] - 50/2 * q[1] ** 2
    nb[2] = 5000 * q[2] - 6/3 * q[2] ** 3
    nb[3] = 1000 * q[3] - 150*8/9 * q[3] ** (9/8)
    nb[4] = 3000 * q[4] - 40/2 * q[4] ** 2
    return -nb.sum()

## import minimize function, along with Bounds and LinearConstraint classes
from scipy.optimize import Bounds, LinearConstraint, minimize

## Set bounds for each q variable
bounds = Bounds([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf])

## set other linear constraints. THis is how we write "0 <= q_a + q_b + q_c + q_d + q_e <= 2000"
linear_constraint = LinearConstraint([1, 1, 1, 1, 1], 0, 2000)

sol_min = minimize(MinusNetBenefits, np.array([400, 300, 500, 600, 500]), bounds=bounds, constraints=linear_constraint, method='trust-constr')
print(sol_min)
print()

print(sol_min['x'] - q_allocation)
print(sol_min)

sol_min.x.sum()

q_allocation = sol_min.x
print(q_allocation)
print()

print(q_allocation.sum())

bounds = Bounds([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, 1500, np.inf])
linear_constraint = LinearConstraint([1, 1, 1, 1, 1], 0, 2000)

sol_min = minimize(MinusNetBenefits, np.array([120, 125, 134, 1500, 101]), bounds=bounds, constraints=linear_constraint, method='trust-constr')
print(sol_min)
print()

q_allocation = sol_min.x
print(q_allocation)
print()

print(q_allocation.sum())

## we kind of did a lot of work for that though, so let's use the quad function
## use scipy integration and constrained optimization approach
from scipy.integrate import quad 

## First, write each marginal net benefits as its own function
def mnb_a(x): 
    return 3000 - 85 * x
def mnb_b(x): 
    return 1500 * 5 - 50 * x
def mnb_c(x): 
    return 500 * 10 - 6 * x ** 2
def mnb_d(x): 
    return 1000 - 150 * x ** (1/8)
def mnb_e(x): 
    return 3000 - 40 * x

## Now write the net benefits as the integral of mnb function
def nb_s(fn, x): 
    return quad(fn, 0, x)[0]

def MinusNetBenefits_integrate(q):
    nb = np.zeros(5)
    nb[0] = nb_s(mnb_a, q[0])
    nb[1] = nb_s(mnb_b, q[1])
    nb[2] = nb_s(mnb_c, q[2])
    nb[3] = nb_s(mnb_d, q[3])
    nb[4] = nb_s(mnb_e, q[4])
    return -nb.sum()

sol_min = minimize(MinusNetBenefits_integrate, np.array([120, 125, 134, 1500, 101]), bounds=bounds, constraints=linear_constraint, method='trust-constr')
print(sol_min)
print(sol_min.x.sum())
x = sol_min.x






























