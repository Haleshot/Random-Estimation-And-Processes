#!/usr/bin/env python
# coding: utf-8

import numpy as np

# Dot and Cross product of vectors
v1 = np.array([1, 2, 3])
v2 = np.array([2, 3, -5])
print ("The Value of Dot product of the two vectors is : ", np.dot(v1, v2))
print ("The Value of Cross product of the two vectors is :\n", np.cross(v1, v2))

# Scalar Multiplication
c = 3 * v1 - 5 * v2
print("The Scalar Multiplication Value is : ", c)

# Defining Matrices
A = np.matrix([[1, 2], [3, 4]])
B = np.matrix([[1, -4], [7, 1]])


print("The Matrix A is :\n", A)
# A.shape gives number of rows and columns
print("\nThe number of Rows and Columns are :", A.shape)

# A.shape gives number of Entries in the Matrix
print("\nThe number of Entries in the Matrix is :", A.size)

print("\nThe Transpose of the Matrix A is :\n", np.transpose(A))
print("\nThe Rank of the Matrix A is :\n", np.linalg.matrix_rank(A))

# Adding two Matrices
print("\n1. The Addition of the Matrices is :\n", np.mat(A) + np.mat(B))
print("\n2. The Addition of the Matrices is :\n", np.mat(B) + np.mat(A))
print("Hence the Addition of Two Matrices is Commutative")

# Mutliplying two Matrices
print("\n1. The Product of the Matrices is :\n", np.mat(A) * np.mat(B))
print("\n2. The Product of the Matrices is :\n", np.mat(B) * np.mat(A))
print("Hence the Addition of Two Matrices is not Commutative")

# Conditions to check
# 1.
print("A + B Transpose:\n")

value_1 = np.mat(A) + np.mat(B)
print(np.transpose(value_1))

print("\n")

print("A Transpose + B Transpose:\n")
value_2 = np.transpose(A) + np.transpose(B)
print(np.transpose(value_1))

print("Hence the first condition is satisfied")

# 2.
print("\n")
print("A Transpose * A:\n")

value_1 = np.transpose(A) * np.mat(A)
print(value_1)

print("\n")

print("A * A Transpose:\n")
value_2 = np.mat(A) * np.transpose(A)
print(value_1)

print("Hence the second condition is satisfied")

#3.
X = A + np.transpose(A)

#4.
Y = A - np.transpose(A)

#5. Check A = 1/2[(X + Y)]
print("\nThe value of 1/2 * [(X + Y)] is :\n", 0.5*(X + Y))


# In[41]:


np.linalg.matrix_rank(A)


# In[42]:


np.trace(A)


# In[61]:


# Mean, Variance and Mode
import statistics as st
x1 = [1, 2, 3, 4]
print(x1)

x2 = [1, 5, -3, 9]
print(x2)

print("Mean = ", np.mean(x1))
print("Mode = ", st.mode(x1))
print("Median = ", st.median(x1))
print("P Variance = ", st.pvariance(x1))
print("Standard Deviation = ", st.pstdev(x1))
print("Variance = ", st.variance(x1))
# Co-variance command is very important for Exam
print("\nCo Variance between x1 and x2 :\n", np.cov(x1, x2))

# Correlation command
print("\nCo relation between x1 and x2 :\n", np.corrcoef(x1, x2))


# In[ ]:




