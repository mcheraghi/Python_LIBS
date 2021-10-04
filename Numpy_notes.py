import numpy as np


#--------------------------------------------------------------------------
my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])

# my_house greater than 18.5 or smaller than 10
a = np.logical_or(my_house>18.5,my_house<10)

# Both my_house and your_house smaller than 11
b = np.logical_and(my_house>18.5,my_house<10)
 



#------------------------------------------------------------------------------------ dict.items(), np.nditer(array)
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin',
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw', 'austria':'vienna' }
          
# Iterate over europe
for a,b in europe.items():
    print("the capital of {} is {}".format(a,b))
    
for x in np.nditer(x): # x is a nd array
	print(x)

