#!/usr/bin/env python
# coding: utf-8

# In[125]:


# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
from sympy import *


# In[40]:


"""
Target for today:
1. Plotting a graph for t = sin(t), where t varies from -2π to 2π using matplotlib and numpy
2. Generate a sinusoidal signal. Plot its ACF.
"""


# In[75]:


# 1. 1. Plotting a graph for t = sin(t), where t varies from -2π to 2π using matplotlib and numpy
 
x = np.linspace(-2 * np.pi, 2 * np.pi, 100) # Plotting the graph for t = sin(t) where t varies
# from -2π to 2π; 100 denotes Linearly spaced numbers
t = np.sin(x)

plt.title("Plotting t = sin(x), were t varies from -2π to 2π")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")

plt.plot(t, marker = 'o')
plt.show()


# In[79]:


x = np.linspace(-np.pi, np.pi, 100) # Plotting the graph for t = sin(t) where t varies
# from -π to π; 100 denotes Linearly spaced numbers
t = np.sin(x)

plt.title("Plotting t = sin(x), were t varies from -π to π")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")

plt.plot(t)
plt.show()


# In[32]:


x = np.linspace(-np.pi/2, np.pi/2, 100) # Plotting the graph for t = sin(t) where t varies
# from -π/2 to π/2; 100 denotes Linearly spaced numbers
t = np.sin(x)


plt.title("Plotting t = sin(x), were t varies from -π/2 to π/2")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.plot(t)
plt.show()


# In[50]:


# 2. Generate a sinusoidal signal. Plot its ACF.
def find_root(a, b, c):
    # The j in our solution stands for i (iota - complex numbers)
    # Calclulate Discriminant
    D = b**2 - (4*a*c)
    print("Discriminant Value = ", D)
    if D > 0:
        print("Roots are real and unique")
        x1 = (-b + cmath.sqrt(D))/(2 * a)
        x2 = (-b - cmath.sqrt(D))/(2 * a)
        print("The Roots are of the equation are :")
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
find_root(1, -5, 6)


# In[48]:


# Sir's version
def find_root(a, b, c):
    D = (b**2) - (4*a*c)
    x1 = (-b + cmath.sqrt(D))/(2 * a)
    x2 = (-b - cmath.sqrt(D))/(2 * a)
    return D, x1, x2
find_root(1, 1, 1)


# In[66]:


# Function to generate a sine wave
def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(freq, sample_rate * duration, endpoint = False)
    frequencies = x * freq
    # 2pi because np.sin takes input in radians
    y = np.sin((2 * np.pi) * frequencies)
    return x, y


# In[87]:


# Generate  a 2 Hertz sine wave that lasts for 5 seconds
SAMPLE_RATE = 441000 # Hertz
DURATION = 5 # seconds
x, y = generate_sine_wave(2, SAMPLE_RATE, DURATION)
plt.plot(x, y, marker = 'o')


# In[70]:


generate_sine_wave(1, 2, 3)


# In[71]:


# Function to generate a cosine wave
def generate_cosine_wave(freq, sample_rate, duration):
    x = np.linspace(freq, sample_rate * duration, endpoint = False)
    frequencies = x * freq
    # 2pi because np.sin takes input in radians
    y = np.cos((2 * np.pi) * frequencies)
    return x, y


# In[81]:


# Generate  a 2 Hertz cosine wave that lasts for 5 seconds
SAMPLE_RATE = 441000 # Hertz
DURATION = 5 # seconds
x, y = generate_cosine_wave(2, SAMPLE_RATE, DURATION)
plt.plot(x, y, marker = 'o')


# In[105]:


nice_tone = generate_sine_wave(400, SAMPLE_RATE, DURATION)
noise_tone = generate_sine_wave(4000, SAMPLE_RATE, DURATION)
mixed_tone = nice_tone + noise_tone

plt.show()


# In[111]:


data = np.array([31.78, 55.65, 44.56])


# In[138]:



data = np.array([24.40, 110.25, 20.05, 22.00, 61.90, 7.80, 15.00, 22.80, 34.90, 57.30])

# Plot autocorrelation
plt.acorr(data, maxlags = 9)


# In[127]:


x, y = symbols('x y')


# In[129]:


integrate(x, x)


# In[130]:


integrate(sin(x) + cos(x), x)


# In[133]:


integrate(x ** 3)


# In[134]:


integrate(exp(x))


# In[135]:


integrate(x, (x, 0, 5)) # Limits of Integral of x from 0 to 5


# In[ ]:




