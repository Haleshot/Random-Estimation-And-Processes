#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Write a python program for a fixed t, in the interval (-3 π to 3 π) for any random process
X(t) = A cos(wt) + B sin(wt)


# In[ ]:


# A = 5 7 420 69 45
# B = 3 4 88  18 77
# w = 1400 2000 3000 9999 21000 


# In[70]:


# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
from sympy import *


# In[ ]:


"""
Target for today:
1. Plotting a graph for X(t) = Acos(wt) + Bsin(wt), where  varies from -3π to 3π using matplotlib and numpy
"""


# In[ ]:


# 1. Plotting a graph for X(t) = Acos(wt) + Bsin(wt), where  varies from -3π to 3π
 
t = np.linspace(-3 * np.pi, 3 * np.pi, 1000) # Plotting the graph for t = sin(t) where t varies

# from -3π to 3π; 1000 denotes Linearly spaced numbers

run = True
while(run):
    f = int(input("Enter the value for Frequency : "))
    A = int(input("Enter the value for A : "))
    B = int(input("Enter the value for B : "))
    w = 1/(2 * np.pi * f

    plot = A * np.cos(w * t) + B * np.sin(w * t)

    plt.title("Plotting X(t) = Acos(wt) + Bsin(wt), were t varies from -3π to 3π")
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")

    plt.plot(plot)
    plt.show()
    run = int(input("Enter 1 for continuing, 0 to quit : "))


# In[ ]:


print(t)


# In[ ]:


print(plot)


# In[32]:


# Integrate 
# 1. f(x) = x^2 from -2 to 5
# 2. f(x) = x^2 * e^-x from 0 to ∞


# In[37]:


x, y = symbols('x y')
integrate(x ** 2, (x, -2, 5)) # Limits of Integral of x from 0 to 5


# In[35]:


integrate(x ** 2 * exp(-x), (x, 0, np.inf)) # Limits of Integral of x from 0 to 5


# In[43]:


# Fourier Transform and Inverse Fourier Transform
s, t, w = symbols('s t w')

"""f(x) = 
        1, |x| < 1
        0, otherwise
"""


# In[54]:


fourier_transform(abs(t), t, w)


# In[55]:


print(fourier_transform)


# In[58]:


# f = -2π * |t|
fourier_transform(-2 * np.pi * abs(t), t, w)


# In[72]:


j = sqrt(-1)
1/sqrt(2*np.pi) * integrate(1 * exp(-j * w * x), (x, -1, 1))


# In[ ]:


# End of the Practical Session

