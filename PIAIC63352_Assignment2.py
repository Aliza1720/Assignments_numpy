#!/usr/bin/env python
# coding: utf-8

# # Numpy_Assignment_2::

# ## Question:1

# ### Convert a 1D array to a 2D array with 2 rows?

# #### Desired output::

# array([[0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9]])

# In[1]:


import numpy as np
arr = np.array([0,1,2,3,4,5,6,7,8,9])
arr.reshape(2,5)


# ## Question:2

# ###  How to stack two arrays vertically?

# #### Desired Output::
array([[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]])
# In[2]:


arr1 = np.array([[0,1,2,3,4], [5,6,7,8,9]])
arr2 = np.array([[1,1,1,1,1], [1,1,1,1,1]])
arr3 = np.vstack((arr1, arr2))
arr3


# ## Question:3

# ### How to stack two arrays horizontally?

# #### Desired Output::
array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1],
       [5, 6, 7, 8, 9, 1, 1, 1, 1, 1]])
# In[3]:


arr1 = np.array([[0,1,2,3,4], [5,6,7,8,9]])
arr2 = np.array([[1,1,1,1,1], [1,1,1,1,1]])
arr3 = np.hstack((arr1, arr2))
arr3


# ## Question:4

# ### How to convert an array of arrays into a flat 1d array?

# #### Desired Output::
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# In[4]:


arr = np.arange(10)
arr.reshape(1,2,5)
arr = arr.flatten()
arr


# ## Question:5

# ### How to Convert higher dimension into one dimension?

# #### Desired Output::
array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
# In[5]:


arry = np.arange(15)
arry.reshape(1,3,5)
arry.flatten()


# ## Question:6

# ### Convert one dimension to higher dimension?

# #### Desired Output::
array([[ 0, 1, 2],
[ 3, 4, 5],
[ 6, 7, 8],
[ 9, 10, 11],
[12, 13, 14]])
# In[6]:


arrr = np.arange(15).reshape(-2,3)
arrr


# ## Question:7

# ### Create 5x5 an array and find the square of an array?

# In[7]:


arr2 = np.arange(25).reshape(1,5,5)
arr2 = np.sqrt(arr2)
arr2


# ## Question:8

# ### Create 5x6 an array and find the mean?

# In[8]:


arr3 = np.random.randint(80, size = (5,6))
print (arr3)
arr4 = np.mean(arr3)
print ('mean is', arr4)


# ## Question:9

# ### Find the standard deviation of the previous array in Q8?

# In[9]:


arr5 = np.std(arr3)
arr5


# ## Question:10

# ### Find the median of the previous array in Q8?

# In[10]:


arr6 = np.median(arr3)
arr6


# ## Question:11

# ### Find the transpose of the previous array in Q8?

# In[11]:


arr7 = np.transpose(arr3)
arr7


# ## Question:12

# ### Create a 4x4 an array and find the sum of diagonal elements?

# In[12]:


arr8 = np.arange(16).reshape(4,4)
print (arr8)
arr9 = np.trace(arr8)
print ("sum of diagnal elemants is" , arr9)


# ## Question:13

# ### Find the determinant of the previous array in Q12?

# In[13]:


arr10 = np.linalg.det(arr8)
arr10


# ## Question:14

# ### Find the 5th and 95th percentile of an array?

# In[14]:


arr11 = np.arange(15)
arr12 = np.percentile(arr11, 5)
print ("5th percentile is", arr12)
arr13 = np.percentile(arr11, 95)
print  ("95th percentile is", arr13)


# ## Question:15

# ### How to find if a given array has any null values?

# In[15]:


arr14 = np.random.randint(0,5, (5,5))
print (arr14)
arr15 = np.any(arr14 == 0)
print ("does this arry contain any null value:" ,arr15)

