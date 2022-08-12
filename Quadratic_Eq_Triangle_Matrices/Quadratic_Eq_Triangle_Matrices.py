#!/usr/bin/env python
# coding: utf-8

# In[12]:


# Algebraic operations
# To run the code press "Shift + Enter"


# In[7]:


13%5


# In[11]:


3 + 6


# In[4]:


12-6


# In[5]:


5/2


# In[6]:


5//2


# In[10]:


5*5


# In[13]:


6-15


# In[17]:


#absolute value (removes the '-' sign)
abs(6-15)


# In[15]:


7%5


# In[3]:


import math
import cmath
import numpy as np
print("MPSTME")


# In[49]:


#x^2-5x+6
# The j in our solution stands for i (iota - complex numbers)
a = 2
b = -5
c = 2
#calclulate discriminant
D = b**2 - (4*a*c)
print("Discriminant Value = ", D)
if D > 0:
    print("Roots are real and unique")
    x1 = (-b + cmath.sqrt(D))/(2 * a)
    x2 = (-b - cmath.sqrt(D))/(2 * a)
    print(x1)
    print(x2)
elif D < 0:
    print("Roots are imaginary")
    x1 = (-b + cmath.sqrt(D))/(2 * a)
    x2 = (-b - cmath.sqrt(D))/(2 * a)
    print(x1)
    print(x2)
elif D == 0:
    print("Roots are equal and real")
    x1 = (-b + cmath.sqrt(D))/(2 * a)
    x2 = (-b - cmath.sqrt(D))/(2 * a)
    print(x1)
    print(x2)


# In[ ]:


# For finding out the square root, we use sqrt function.
import math
math.sqrt(9)


# In[62]:


print("Enter the Sides for the Triangle")
a = int(input("Enter the Value for Side 1 : "))
b = int(input("Enter the Value for Side 1 : "))
c = int(input("Enter the Value for Side 1 : "))
s = (a + b + c)/2
area = math.sqrt((s*(s-a)*(s-b)*(s-c)))
print("The Area of the Triangle is = ", area)


# In[94]:


v1 = np.array([[1, 2], [2, 3]])
v1


# In[92]:


A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = np.array([[134, 522, 234], [344, 5445, 656], [637, 845, 349]])
print(A * B)
print(B * A)


# In[4]:


#Consider two matrices A and B of orders 2x4 and 4x2 and Solve the following
# 1. A * B
# 2. B * A
A = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
B = np.array([[1, 2] , [3, 4], [5, 6], [7, 8]])
print(A * B)
print(B * A)


# In[ ]:




