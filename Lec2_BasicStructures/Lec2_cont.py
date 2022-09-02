# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 16:26:01 2022

@author: rcuppari
"""

#############################################################################
######################### WELCOME TO SPYDER #################################
######### an absolutely marvelous interface for coding in Python ############
#############################################################################

## TODAY: intro to Spyder, finishing up basic structures & introducing if statements 

## Two more structures: tuples and dictionaries 

print("A tuple has () instead of [], but we still access it through []")

my_list = [9, 1, 2022]
my_tuple = (9, 1, 2022)
#print(type(my_tuple))
#print(my_tuple[0])

## who read ahead and knows the difference between a tuple and a list? 
#my_tuple[1] = 2
#my_list[1] = 2

#my_tuple = (9, 2, 2022)

print("Dictionaries are useful when you want to store multiple sets of data together")
Gillings = {'Degree': 'MS', \
            'Program': 'ESE', \
            'Focus': 'Environmental Justice'}
## note: backslashes, commas, colons, and curly braces! 
print(type(Gillings))
print(Gillings.keys())
print(Gillings.values())
print(Gillings.items())

## how do you think we add a key? 

## we can also combine two dictionaries into a list where each list item is 
## a dictionary, or put lists inside of dictionaries 
degrees = ['MS', 'MSEE', 'MPH', 'MHA', 'MSCR', 'PhD']
Gillings['Degree'] = degrees
Gillings['Program'] = ['ESE', 'EPID']
Gillings['Degree'] = degrees + degrees
Gillings['Degree'] = [degrees + degrees]
Gillings['Degree'] = [degrees] + [degrees]

###############################################################################
###########################      In-class exercises  ##########################
#Create and print:
#    "t1": a tuple with a single element, "0"
#    "t2": a tuple with the elements 5, 1, -3, -7, -11. Try to do this without typing everything (hint: tuple/range).
#    "d1": a dictionary with two keys, "Fact" and "Fiction". The values for these keys are True and False, respectively.
#    "d2": a dictionary with five keys, "l1", "l2", "t1", "t2", "d1". The value for each key is the corresponding object.
