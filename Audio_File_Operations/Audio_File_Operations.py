#!/usr/bin/env python
# coding: utf-8

# In[1]:


# RPT prac 06 B3 I066 Srihari Thyagarajan 8th September


# In[106]:


"""
Consider 2 signals, say signal 1 and signal 2, with frequency 50Hz and 150Hz in the interval (-2π t 2π).

a) Signal 1
b) Signal 2
c) Signal 1 + Signal 2
d) Signal 1 - Signal 2
Interpret the result.

"""


# In[150]:


# import libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import math


# importing libraries for playing sound file
from scipy.io.wavfile import read,write
from IPython.display import Audio
from numpy.fft import fft, ifft
from playsound import playsound # This Library is used.
get_ipython().run_line_magic('matplotlib', 'inline')


# In[104]:


# Declaring respective variables
F1 = 50
F2 = 150


# In[152]:


# Comparing 2 Signals
t = np.linspace(-2 * math.pi, 2 * math.pi, 50)
x1 = np.sin(F1 * 2 * math.pi * t)

plt.title("Comparing 2 Signals, where Frequency varies from 50 to 150 and interval lies between -2π to 2π")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.legend("Signal 1")
plt.plot(t, x1)

x2 = np.sin(F2 * 2 * math.pi * t)
plt.plot(t, x2)


# In[153]:


# Plotting the 2 signals -
# signal 1 + signal 2

plt.title("Comparing 2 Signals, Signal 1 + Signal 2")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.legend("Signal 2")

plt.plot(t, x1 + x2)


# In[98]:


# signal 1 - signal 2
plt.title("Comparing 2 Signals, Signal 1 - Signal 2")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.legend("Signal 3")
plt.plot(t, x1 - x2)


# In[131]:


from playsound import playsound
playsound('C:/Users/mpstme.student/Downloads/I066_Folder/Experiment_8/audio.wav')
print('playing sound using  playsound')


# In[149]:


Fs, data = read('C:/Users/mpstme.student/Downloads/I066_Folder/Experiment_8/audio.wav')
data = data[:,0]
print(Fs)


# In[ ]:




