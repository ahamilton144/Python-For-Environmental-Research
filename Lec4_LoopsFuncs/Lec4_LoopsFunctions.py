# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 22:58:38 2022

@author: rcuppari
"""


#############################################################################
############# LECTURE 4: LOOPS, FUNCTIONS, and COMPREHENSIONS ###############
#############################################################################

## GOALS: learning about for loops and using an if-else statement across 
## multiple values, learning to create a function to easily duplicate the same
## action, and applying the same manipulation/action across an entire list  

###############################################################################
############################## FOR LOOPS!! ####################################
## for when we want to do the same task over and over and over again 

## we can tell the computer what to loop - or iterate - over, 
## for example, a RANGE of numbers: 
for i in range(10): 
    print(i)
    
## or items within a list
elements = ['earth', 'air', 'fire', 'water']
for e in elements: 
    print(e)

##########################################
## so what does the i or e stand for here? 

## understanding that ^, we can consider a situation in which we want both 
## the number item and the value of the item: 
for e, element in enumerate(elements): 
    print("Element #" + str(e+1) + " is " + element)

## a loop could be useful for adding items to a list: 
my_list = []
for i in range(1, 11):
    square = i **2
    my_list.append(square)
print(my_list)    

#################### a hesitant lesson: while loops ###########################
## for when we want to do the same task UNTIL something happens 
## or while something does not happen (maybe because you don't have a set 
## number of iterations)
count = 0
while count < 10: 
    print(count)
    count += 1 
#    count+1

########################### IN CLASS EXERCISE #################################
## create three lists 50 items long using a while loop: "time", "day", & "accident"
## I've already given you an example for the first two lists
## now do the same for accident 
time = []
day = []
accident = []
while ## FILL IN 
# HINT!    count += 1 
    new_time = random.choices(list(range(0,24)), weights = [(1/24)]*24)[0]
    time.append(new_time)
    
    new_day = random.choices(["weekend", "weekday"], weights = [2/7, 5/7])[0]
    day.append(day)
    
    ## add in accident list 
    
## now, create a for loop that will iterate over your two lists
## and then print what sort of traffic to expect at each point
from PIL import Image
image = Image.open('tree_traffic.jpg')
image.show()

for i in : ## FILL IN!
    if day == ## FILL: 
    
        ## move to time of day 
        
            ## move to accident or no accident
    else: 

        ## move to time of day 
        
            ## move to accident or no accident

###############################################################################
######################### LIST COMPREHENSIONS!! ###############################
## list comprehensions are basically more compact for loops, specifically for lists 

## maybe we want to see how many households in NC have private wells 
## let's make up some random data 
import random
## numeric index for each household in county
household = ['H' + str(i) for i in list(range(1000))]
# print(household)

water = [random.choices(['municipal', 'private'], weights = [0.6, 0.4], k=1)[0] for h in household]
# print(water)

## we can use a for loop to see how many private wells there are
private_hh = []
for i in range(1000):
    if water[i] == 'private':
#         private_hh.append('H'+str(i))
        private_hh.append(household[i])
print(private_hh)

## or we can create a list that contains all of the values
## that are equal to "private"
private_hh = [household[i] for i in range(1000) if water[i] == 'private']
len(private_hh)


## now, let's pretend we are in charge of subsidies to 1000 households with private wells
## only below an income threshold

## randomly assign income for each household from Normal/Gaussian distribution
income = [max(random.gauss(50000, 20000), 0) for h in household]
# print(income)

## we can use list comprehensions for the multiple conditions: 
## a household must have a private well AND income < 30,000

eligible = [household[i] for i in range(len(household)) if (water[i] == 'private') and (income[i] < 30000) ]
print(len(eligible))

########################### IN CLASS EXERCISE #################################
## create a list comprehension for eligibility based on whether a household
## has private water AND household # is < 250  


###############################################################################
############################ FUNCTIONS!! ######################################
## functions take a set of inputs, performs an operation, and spits out the 
## output of the operation

## we can set a default
def hello_name(name = 'Rosa'): 
    print("Hello " + name + "!")

hello_name()

hello_name("Santa")


## we can also not set a default
def hello_name_age(name, age):
    s = f'Hello {name}, you are {age:,} years old!'
    print(s)

hello_name_age()

## we can make pretty complex functions, and retrieve specific outputs 
## without printing them 
def fibonacci_under_threshold(threshold):
    fib = [0, 1]

    ## perform while loop until we pass threshold, adding each to list
    continue_loop = True
    while continue_loop:
        new_fib = fib[-1] + fib[-2]
        if (new_fib < threshold):
            fib.append(new_fib)
        else:
            continue_loop = False
    
    return fib

f200 = fibonacci_under_threshold(200)
print(f200)
print()

f1000 = fibonacci_under_threshold(1000)
print(f1000)

## sometimes we think a function isn't working because we see nothing... 
def hello_name(name = 'Rosa'): 
#    print("Hello " + name + "!")
    name2 = name + '2'
hello_name()


## functions are a little complex because of "scope", which is to say that 
## variables defined within functions are not saved for the rest of your 
## script
import math
def hypotenuse(side_a, side_b):
    side_c = math.sqrt(side_a **2 + side_b **2)
    return side_c

hypotenuse(3,4)
print(side_c)
print(side_a)

side_c = hypotenuse(3,4)
print(side_c)

## to that end, alterations to a variable within a function are merely temporary
x = 3
def square(x):
    x **= 2
    return x
y = square(x)
print(x, y)

## At the same time, if we do not define a variable we use in a function 
## right within that function or as an argument, Python will look for that 
## variable in the previously executed lines of code 
x = 3
def multiply_by_x(y):
    return y * x
z = multiply_by_x(4)

print(x, z)

########################### IN CLASS EXERCISE #################################
###################### list comprehension + function #########################

## make a simple function that takes an input of temperature in celsius
## and returns fahrenheit (conversion:( C * 9/5) + 32)

## create a print statement to show that it works when you input zero degrees
func_output = ## FILL IN 
print('Zero degrees Celsius is equal to ' + func_output + \
      ' degrees Fahrenheit')

## now do this  for 0 - 40 degrees  
converted = []
for t in # FILL IN
    deg_fah = # FILL IN 
    converted # FILL IN
print(converted)

## do this with a list comprehension