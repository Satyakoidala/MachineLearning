#!/usr/bin/env python
# coding: utf-8

# # Exceptions

# In[165]:


try : 
    #a = int('a') # Value Error
    #a = 100/0 # ZeroDivionError

    #l = [1,2,3]
    #print(l[5]) # Index out of range intentionally
    #f = open("fileerror.txt", "r") # FileNotFoundError
    pass
except (ValueError, ZeroDivisionError) as e:
    print("*** Error Non numeric value given, give numeric value as input", e)
except (ZeroDivisionError) as e :
    print("*** Error Divide by zero", e)
except (IndexError) as e :
    print("Index error", e)
except (FileNotFoundError) as e :
    print("file does not exist", e)
except Exception as e :
    print("From exception block\n", e.__class__.__name__, e)


# In[153]:


# No exception handling

def exceptionFunc1() :
    exceptionFunc2()

def exceptionFunc2() :
    i = int('abc')

exceptionFunc1()


# In[155]:


# Simple exception handling

def exceptionFunc1() :
    exceptionFunc2()

def exceptionFunc2() :
    i = int('abc')

try : 
    exceptionFunc1()
except Exception as e :
    print("From exception block", e)


# In[152]:


# With stack trace
import sys, traceback

def exceptionFunc1() :
    exceptionFunc2()

def exceptionFunc2() :
    i = int('abc')

try :
    exceptionFunc1()
except Exception as e :
    print("Exception in user code:")
    print("-"*60)
    traceback.print_exc(file=sys.stdout)
    print("-"*60)


# In[241]:


import os


# In[ ]:


# Checking for file.

if not os.path.isfile("dummyfile.txt") :
    print("Error dummyfile.txt does not exist")
    
# chekcing for folder
if not os.path.isdirs("/hallmark")
    print("Error /hallmark does not exist")

# Splitting filename to path, filename prefix, extension
fullname = "/users/python/data.txt"
filenameext = os.path.basename(fullname)
path = fullname.replace(filenameext, '')
filename, ext = os.path.splitext(filenameext)


# In[7]:



# Writing to file
lines = '''Over the course of my working life I have had the opportunity to use 
many programming concepts and technologies to do countless things. 
Some of these things involve relatively low value fruits of my labor, 
such as automating the error prone or mundane like report generation, task 
automation, and general data reformatting. Others have been much more valuable, 
such as developing data products, web applications, and data analysis and 
processing pipelines. One thing that is notable about nearly all of these 
projects is the need to simply open a file, parse its contents, 
and do something with it.

However, what do you do when the file you are trying to consume is 
quite large? Say the file is several GB of data or larger? Again, 
this has been another frequent aspect of my programming career, 
which has primarily been spent in the BioTech sector where files, 
up to a TB in size, are quite common.

The answer to this problem is to read in chunks of a file at a time, 
process it, then free it from memory so you can pull in and process 
another chunk until the whole massive file has been processed. 
While it is up to the programmer to determine a suitable chunk 
size, perhaps the most commonly used is simply a line of a file at a time.
'''

# Opening file and file modes, r, w, a, rb, wb
f = open("dummyfile.txt", "w")
f.write(lines)
f.close()


# In[8]:


# Read all file contents at once
f = open("dummyfile.txt", "r")
data = f.read()

lines = data.splitlines()
for line in lines : 
    print('first', line)
f.close()


# In[9]:


# Read line by line
f = open("dummyfile.txt", "r")

line = f.readline()
count=0
while line :
    count+=1
    print(count, line) # You need to remove newline at end of line.
    line = f.readline()
f.close()


# In[10]:



# More elegant file read
with open("dummyfile.txt", "r") as f :
  for count, line in enumerate(f) :
      print(count, line)


# In[13]:


# Writing to files using write, writelines
lines = ["line1", "line2", "line3"]
with open("dummyfile1.txt", "w") as f :
    lines = ["line1", "line2", "line3"]
    lines = [line + "\n" for line in lines]
    f.writelines(lines) # Does not add end of line

f = open("dummyfile1.txt", "r")
data=f.read()
print(data)
f.close()


# # Function Programming

# In[3]:


def odd(x) :
    return x % 2 != 0

print(list(filter(odd, range(10))))


# In[4]:


def square(x) :
    return x**2

print(list(map(square, range(10))))


# In[5]:


from functools import reduce

def mySum(x,y) :
    return x + y

print(reduce(mySum, range(10)))


# # Reading/Writing JSON From/To file.  

# In[14]:


import json


# In[16]:


key = ['a', 'b', 'c', 'd', 'e']
value = [1,2,3,4,5]
d = dict(zip(key, value))
print(d)


# In[17]:


s = json.dumps(d)
print(s)


# In[18]:


s1 = json.loads(s)
print(s1)


# In[20]:


with open("jsondump.dat", "w") as f:
    json.dump(d, f)

with open("jsondump.dat", "r") as f:
    dnew = json.load(f)
print(dnew)    


# # Date times to Strings, Strings to Date times

# In[34]:


# Python date times

from datetime import datetime

d = datetime.now()
print(d)

d = datetime.strptime("19-12-10 09:35:11", "%y-%m-%d %H:%M:%S")
print(d)

print(datetime.strftime(d, "%Y-%m-%d %H:%M:%S"))

print(d.year, d.month, d.day, d.hour, d.minute, d.second)


# # Date Time arithmetic

# In[45]:


#Date Arithmetic

from datetime import datetime
from datetime import timedelta

d = datetime.strptime("2019-03-31", "%Y-%m-%d")
print(d - timedelta(days=1))
print(d - timedelta(weeks=1))
print(d - timedelta(hours=1))

# Month -1 is not available in timedelta, for that you need to use relativedelta
from dateutil.relativedelta import relativedelta
td = relativedelta(months=1)
print(d - td)

# Substract a year
td = relativedelta(years=1)
print(d - td)


# # Numpy

# In[70]:


import numpy as np

ia = np.array([1,2,3,4], dtype="float64")
print(ia, ia.dtype, ia.shape, ia.size)

ia = np.array([[1,2,3,4],[5,6,7,8]])
print(ia, ia.dtype, ia.shape, ia.size)

# Indexing
print("[1]", ia[1, 0], ia[1][0])

# Slicing
print("2 : ", ia[:2, :2])


# In[58]:


# Numpy data generation, excluding end. params : start, end, step similar to range function
a = np.arange(1,10,0.5)
print(a)


# In[72]:


# Numpy data generation using linspace, start, stop inclusive. params: start, stop, # of numbers to generate
a = np.linspace(1,10, 5)
print(a)


# In[73]:


l = np.zeros((4,4))
print(l)


# In[80]:


l = np.ones((4,4))
print(l*10)

# * on regular lists behaves differently
l = [1,2]
print(l*10)


# In[76]:


l = np.eye((4), dtype='int')
print(l)


# In[244]:


np.random.seed(100)

# Generate 5 random numbers between 0 and 1
x = np.random.rand(5)
print("random.rand : ", x, x.mean(), x.var())

# Generate 5 floating point numbers from Standard normal distribution with mean=0, var=1
x = np.random.randn(5)
print("random.randn : ", x, x.mean(), x.var())

# Generate (2,2) random integers using digits between 10 and 20.
x = np.random.randint(10,20,(2,2))
print("randon.randint : \n", x)


# In[119]:


import statistics

# Generate 10 floating point numbers from normal distribution with mean=5, std=1
x = np.random.normal(loc=5, scale=1, size=10)
print(x, "\nmean : ", x.mean(), x.var(), statistics.stdev(x))


# In[123]:


np.random.seed(42)
print(np.random.randint(100)) # Genrate a random int under 100
print(np.random.randint(1,100,10)) # Generate 10 random ints between 1 and 100


# In[149]:


# Numpy Reshape
na = np.random.randint(1,20,12)
print(na)
na = na.reshape((4,3))
print(na)

# Aggregate across the array 
print("max : ", na.max())
print("sum : ", na.sum())
print("min : ", na.min())

# Aggregate across axis=0, along columns, returns one value per column
print("max : ", na.max(axis=0), na.argmax(axis=1))
print("sum : ", na.sum(axis=0), na.argmax(axis=1))
print("min : ", na.min(axis=0), na.argmax(axis=1))

# Aggregate across axis=1, along row, returns one value per row
print("max : ", na.max(axis=1), na.argmax(axis=1))
print("sum : ", na.sum(axis=1), na.argmax(axis=1))
print("min : ", na.min(axis=1), na.argmax(axis=1))


# In[168]:


a = np.arange(20).reshape((4,5))
print(a)
np.random.shuffle(a) # Shuffles numpy array in place
print("\nAfter Shuffle : \n", a)


# In[173]:


print(a)
a[1] = a[1]+10
print(a)

# Bulk Assignment
a[:2] = 100 
print(a)


# # Pandas - PANel DAta

# ## Pandas - Series

# In[176]:


import pandas as pd

names = ['John', 'Jeff', 'Sue', 'Ann']
s =  pd.Series(data=names, index=['a', 'b', 'c', 'd'])
print(s)


# In[178]:


s1 = s.copy()


# In[186]:


s1.set_value('f', 'Joe')


# In[192]:


s['b':]   # Slicing or single row selection, single set of brackets


# In[195]:


s[['a', 'b']] # Multiple row selection, double square brackets


# ## Pandas - Data Frames
# ### Multiple Series with same index

# In[198]:


from numpy.random import randn
np.random.seed(101)
rand_mat = randn(5,4)
print(rand_mat)


# In[201]:


df = pd.DataFrame(data=rand_mat)
print(df)


# In[223]:


df = pd.DataFrame(data=rand_mat, index='a b c d e'.split(), columns='w x y z'.split())
print(df)
df['new_index'] = ['aa', 'bb', 'cc', 'dd', 'ee']
print(df)
df.reset_index(inplace=True)
print(df)
df.set_index('new_index', inplace=True)
print(df)


# In[234]:


# To access a column
print(df['index'])

# To access a set of columns
print(df[['w', 'x']])

# To Access using indexes
print(df.iloc[2:, 2:])

# To Access using index, column names
print(df.loc[['aa', 'bb'],['w', 'x']])


# In[221]:


f = open("dataForPandas.csv", "w")
f.write("""TranDate,Item,Amount
2019-10-01,1100
2019-10-02,1200
2019-10-03,1150
2019-10-04,1200
2019-10-05,1175
2019-10-06,1240
""")
f.close()
df = pd.read_csv('dataForPandas.csv',index_col='TranDate',parse_dates=True)
df.index.freq = 'D'
get_ipython().run_line_magic('matplotlib', 'inline')
df.plot(figsize=(12,8))


# In[ ]:




