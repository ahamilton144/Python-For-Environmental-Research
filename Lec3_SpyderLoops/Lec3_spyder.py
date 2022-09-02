# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 16:47:53 2022

@author: rcuppari
"""

#############################################################################
################ LECTURE 3: CONDITIONS AND LOOPS ############################
#############################################################################

## GOALS: figuring out how to automate conditional and/or repetitive tasks
## to do that, we need to know how we compare information: 
    
room = 1304
old_room = 1302 
office = '014' # why do y'all think I made this a string? 

## are the values equal? 
print(room == old_room)
print(room == office)
print(room == 1304)

## conversely, can ask whether NOT equal
print(room != old_room)
print(room != office)
print(room != 1304)

## other operators: 
print(9 > 10)
print(9 < 10)
print(9 >= 10)
print(9 <= 10)
print(9 <= 9)

## similarly: 
print(room < 140 or room > 1300)
print(room < 140 and room > 1300)

## so what if we want to analyze data based on water quality and weather, 
# we may know that when the weather is dry then disease goes down 
# and that when weather is wet, we see some spikes 
# and that when there are floods, we see consistent increases 
# we might also need to adjust for population density/km2
# and then we might have to ask whether there is piped sewage or outdoor 
weather = 'wet'
pop_density = 25357 # let's take Mumbai's
sewage = 'piped'

if weather == 'monsoon' and sewage == 'none' and pop_density > 10000:
    print('Watch out for disease spread!')
else: 
    print("No cause for alarm!")

## remember order of operations!
alarmed = weather == 'monsoon' or weather == 'wet' and sewage == 'none'
print(alarmed)
alarmed = sewage == 'none' and weather == 'monsoon' or weather == 'wet'
print(alarmed)

## even better - esp. for future you - use parentheses! 
alarmed = (weather == 'monsoon' or weather == 'wet') and sewage == 'none'
print(alarmed)
alarmed = weather == 'monsoon' or (weather == 'wet' and sewage == 'none')
print(alarmed)

## it can be hard to keep track of multiple conditions though, so we can use 
## "membership" operators  
bad_weather = ['monsoon', 'wet']
x = 'wet'
y = 'monsoon'
print(x in bad_weather)
print(y not in bad_weather)

weather in bad_weather and sewage == 'none'

## now, we might want to nest our if statements... 
age = 27
birthday = "1/7"
today = "9/1"
tomorrow = "9/2"

if today == birthday:
    age += 1
    print( f'Happy birthday, you are now {age}!')
elif tomorrow == birthday:
    age += 1
    print( f'Wow, you turn {age} tomorrow!')
else:
    print( f'Sorry, you are still {age} for at least two days :(')    

## or really nest them
birthday_sadness_threshold = 30
age = 30
today = '1/7'
if age < birthday_sadness_threshold:
    if today == birthday:
        age += 1
        print( f'Happy birthday, you are now {age}!')
    elif tomorrow == birthday:
        age += 1 
        print( f'Wow, you turn {age} tomorrow!')
    else:
        print( f'Sorry, you are still {age} :(')  
else:
    if today == birthday:
        age += 1
        print( f'Ugh, another year around the sun. You are now {age}, gross!')
    elif tomorrow == birthday:
        age += 1
        print( f'Enjoy your last day of being {age - 1}, you turn super old ({age}!) tomorrow.')
    else:
        print( f'Hooray, you are still {age} for at least two days :)')  


########################### IN CLASS EXERCISE #################################
## make an if statement corresponding to the following image: 
from PIL import Image
image = Image.open('tree_traffic.jpg')
image.show()



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
## create a while loop that will keep randomly flipping a coin until 
## a "tails" ("T") occurs. Append each outcome to a list. 
## At that point, print the number of coin flips
import random
HorT = random.choices(['H', 'T'], weights = [0.9, 0.1])[0]

## now let's do this in a for loop, so that you flip your figurative coin
## 100 times and store all of the values











