#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy

arr = numpy.array([1, 2, 3, 4, 5])

print(arr)


# In[2]:


import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

print(arr)


# In[3]:


import numpy as np

arr = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

print(arr)


# In[4]:


import numpy as np

a = np.array(42)
b = np.array([1, 2, 3, 4, 5])
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

print(a.ndim)
print(b.ndim)
print(c.ndim)
print(d.ndim)


# In[5]:


import numpy as np

arr = np.array([1, 2, 3, 4])

print(arr[0])


# In[6]:


import numpy as np

arr = np.array([1, 2, 3, 4])

print(arr[2] + arr[3])


# In[7]:


import numpy as np

arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])

print('2nd element on 1st row: ', arr[0, 1])


# In[8]:


import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[1:5])


# In[9]:


import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[4:])


# In[10]:


import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[:4])


# In[11]:


import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[-3:-1])


# In[13]:


import numpy as np

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

print(arr.shape)


# In[14]:


import numpy as np

arr = np.array([3, 2, 0, 1])

print(np.sort(arr))


# In[15]:


x = 5
y = "nodi"
print(x)
print(y)


# In[18]:


x = 5
y = "nodi"
print(type(x))
print(type(y))


# In[19]:


thislist = ["apple", "banana", "cherry"]
print(thislist)


# In[20]:


thislist = ["apple", "banana", "cherry"]
print(len(thislist))


# In[21]:


mylist = ["apple", "banana", "cherry"]
print(type(mylist))


# In[22]:


thislist = ["orange", "mango", "kiwi", "pineapple", "banana"]
thislist.sort()
print(thislist)


# In[23]:


thislist = [100, 50, 65, 82, 23]
thislist.sort()
print(thislist)


# In[24]:


thislist = ["orange", "mango", "kiwi", "pineapple", "banana"]
thislist.sort(reverse = True)
print(thislist)


# In[25]:


a = 33
b = 33
if b > a:
  print("b is greater than a")
elif a == b:
  print("a and b are equal")


# In[26]:


fruits = ["apple", "banana", "cherry"]
for x in fruits:
  print(x)
  if x == "banana":
    break


# In[27]:


i = 1
while i < 6:
  print(i)
  if i == 3:
    break
  i += 1


# In[ ]:




