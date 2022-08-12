#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[69]:


# 1. Histogram Distribution
# 2. Bernoulli Distribution
# 3. Uniform Distribution
# 4. Normal Distribution
# 5. ACF Plot (if possible)


# In[139]:


# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
from scipy.stats import uniform
from scipy.stats import norm
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf


# In[79]:


x = np.array([10, 35, 26, 35, 4, 5, 2, 17, 5, 4, 37, 26, 200, 255, 259, 201, 301, 189, 35, 45, 54, 4, 7, 38, 54, 34, 73, 33, 72, 54, 44, 23, 32, 34, 356, 36, 35, 34, 35, 2, 6, 7, 3, 2, 3, 2])
x


# In[80]:


# Plotting a Histogram
plt.hist(x)
plt.xlabel("X Axis") # Label for X Axis
plt.ylabel("Y Axis") # Label for Y Axis
plt.title("Historgram") # Label for the Histogram
# plt.legend("x") # Legend key for the Histogram
plt.savefig('hist01.png') # Command for saving the Historgram as a png in the same directory in Jupyter
plt.show() # To display the Histogram


# In[81]:


#  Bernoulli Distribution for an Far coin
num_tosses = 1000
# p = 0.5 is for fair coin, any other value of p results in unfair coin
fair_coin = bernoulli.rvs(p = 0.5, size = num_tosses) # We try to get a bernoulli distribution for a fair coin with 0.5p with 1000 tosses
plt.hist(fair_coin) # We try to plot a histogram for a fair coin with 1000 tosses

plt.title('Bernoulli Distribution for a fair coin')


# In[84]:


#  Bernoulli Distribution for an Unfair coin
num_tosses = 1000
# p = 0.3 is for an unfair coin, any other value of p results in unfair coin
unfair_coin = bernoulli.rvs(p = 0.3, size = num_tosses) # We try to get a bernoulli distribution for a fair coin with 0.5p with 1000 tosses
plt.hist(unfair_coin) # We try to plot a histogram for a fair coin with 1000 tosses

plt.title('Bernoulli Distribution for an Unfair coin')


# In[119]:


# Uniform Distribution
uniform_rv1 = uniform.rvs(size = 1000)
plt.hist(uniform_rv1, bins = 50, density = 1)
plt.title("Uniform Distribution")
plt.xlabel("Value of Uniform RV")
plt.ylabel("Relative Frequency of Occurence")
plt.show()


# In[116]:


# Generating a Gaussian Random Variable having 50 samples
norm_rv1 = norm.rvs(size = 50)
print(norm_rv1)
print("The Mean of the Normal RV1 is = %0.3f" % np.mean(norm_rv1)) # 0.3f -> For 3 decimal places
print("The Standard Deviation of Normal RV2 is = %0.3f" % np.std(norm_rv1))

# You can also print the above statements in this order : 
# print("The Mean of the Normal RV1 is = ", round(np.mean(norm_rv1), 3))
# print("The Standard Deviation of Normal RV2 is = ", round(np.std(norm_rv1), 3))


# In[120]:


# Plotting the Distribution of the Generated Random Variable
plt.hist(norm_rv1)
plt.title("Normal (Gaussian) Distribution")
plt.xlabel("Value of Gaussian RV")
plt.ylabel("Frequency of Occurence")
plt.show()


# In[137]:


# Auto Correlation Function Plots (ACF plots)
data = pd.read_csv('C:/Users/mpstme.student/Downloads/I066_Folder/Experiment_4/AAPL.csv') # Reading of the Apple Stock History stored on the Computer
data


# In[145]:


data_1 = data[["Date", "Adj Close"]].set_index(["Date"])
data_1


# In[149]:


plot_acf(data_1)


# In[ ]:




