# -*- coding: utf-8 -*-
"""notebooks_RPT prac 03 B3 Srihari Thyagarajan 28th July.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_F9f6xYOeSxMo99Zzm_kdveBcCVqOoH2

# prac 03 B3 Srihari Thyagarajan 28th July

![image.png](attachment:d182ceca-488d-4cf2-96b1-59f83eba0d61.png)
"""

# All import modules are declared in this cell.
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 200)
#print(x)

y = x**2

plt.plot(x, y)
plt.show()

x3 = np.linspace(-5, 5, 3)

y = x3**2
plt.plot(x3, y)
plt.show()

x9 = np.linspace(-5, 5, 9)

y = x9**2
plt.plot(x9, y)
plt.show()

x200 = np.linspace(-5, 5, 200)

y = x200**2
plt.plot(x200, y)
plt.show()

x2000 = np.linspace(-5, 5, 2000)

y = x2000**2
plt.plot(x2000, y)
plt.show()

x3 = np.linspace(-5, 5, 20)

y = x3**2
plt.plot(x, y)
plt.scatter(x3, y)
plt.show()

# Formatin of a matrix whose Eigen Values and Vectors are to be obtained.
A = np.matrix([[2, 4, 5], [6, 3, 8], [7, 34, 2]])
print(A)

# We get the Eigen Value and Vector of the Matrix A with the following line:
np.linalg.eig(A)
## We get the Eigen Value and Vector of the Matrix A individually with the following line:
[eigval, eigvec] = np.linalg.eig(A)

# To get the Eigen Values of the matrix
eigval

# To get the Eigen Vector of the matrix
eigvec

# To determine eigen value at a particular index.
eigval[1]

import pandas as pd # For reading the CSV File

data = pd.read_csv('F:/New_Laptop_Documents/NMIMS_College_Docs/2nd_Year/1st_Semester/RPT/Practicals/Exp_3/AMZ1.csv')

