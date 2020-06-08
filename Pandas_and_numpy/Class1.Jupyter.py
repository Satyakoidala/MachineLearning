#!/usr/bin/env python
# coding: utf-8

# In[30]:


print("Hello World")


# In[31]:


# String example
s = '123456789'
dir(s)

# Check if string is numeric, if so, convert it to integer
s = '100'
print(s, type(s))
if s.isnumeric() :
    s = int(s)
else :
    print("not numeric")
print(s, type(s))


# In[33]:


# Different forms of quotes 
s="I like python"
s='I like python'
s = """I like python
it is very versatile"""


# In[34]:


# Boolean, note: no quotes and first char upper case
b = True
c = False
print(b, c, type(b))
if not b :
    pass


# In[35]:


# If with equality operator
x = 100
if x == 100 :
    print("100")
else :
    print("not 100")


# In[36]:


# String Operations, slicing
s = '0123456789'
print(s[1])
print(s[-1])
print(s[:5],s[5:])
print(s[::2])
print(s[-5:])
print(s[:-3])
print(s[::-1])
print(s[::-2])


# In[37]:


# List access is similar to string, slicing behaves same.
s = [0,1,2,3,4,5,6,7,8,9]
print(s[1])
print(s[-1])
print(s[:5],s[5:])
print(s[::2])
print(s[-5:])
print(s[:-3])
print(s[::-1])
print(s[::-2])


# In[38]:


# List with different types of elements
l = ['string', 100, True, 200.0, [1,2,3]]
print(l)
print(l[0], l[1], l[2], l[3], l[4], l[4][1])
dir(l)


# In[39]:


# Tuple is a read only copy of list
t = (1,2,3,4,5)
print(t)
# another way to create tuple from a list
t = tuple([1,2,3,4,5])
print(t)


# In[40]:


l = [1,2,3,4,5,1,2,3,1,2]
# Create set from list l, Sets use curly braces
s = set(l)
print(s)


# In[41]:


# Try to access set with index, it gives error
s[0]


# In[42]:


# list functions on set s
dir(s)
# Add delete elements from s
s.add(5)
print(s)
s.pop()
print(s)


# In[43]:


# Create list and tuple from set
l1 = list(s)
print(l1)
t1 = tuple(s)
print(t1)
print(s)
s1 = {4,5,6,7}
print(s)
print(s1)


# In[46]:


# Set operations on sets
s.union(s1)
dir(s)
print(s.difference(s1))
print(s.intersection(s1))


# In[47]:


# Range Function to generate numbers
range(10)
print(list(range(10)))
print(list(range(10,20)))
print(list(range(10,20,2)))


# In[48]:


#Define Dictionary
d = {'fname' : 'Ravi', 'lname' : 'Y'}

#Access Dictionary
print(d['fname'])

# Functions on dictionary
dir(d)

print(d)


# In[49]:


# Get help on any python function/object
help(d.get)


# In[50]:


# Update vlaues in dictionary
d['hobbies'] = ['reading', 'hiking']

#Access Dictionary
print(d['hobbies'])

d['hobbies'] = ['reading', 'hiking', 'biking']
print(d)


# In[51]:


# Add 100 as new key to d
d[100] = 200
print(d)


# In[52]:


# Remove Key 100 from d
del d[100]
print(d)


# In[53]:


# Check if key is in d
'fname' in d
'abc' in d
'abc' in d.keys()
'fname' in d.keys()
print(d)


# In[54]:


# While loop
l= [1,2,3,4,5]
while len(l) > 0 :
    e = l.pop()
    print(e)


# In[55]:


# For loop with else
for i in range(10) :
    print(i)
else :
    print("at the end of loop")


# In[56]:


# For loop with else, does not go to else because of break
for i in range(10) :
    print(i)
    if i > 5 :
        break
else :
    print("at the end of loop")


# In[58]:


l = [1,2,3,4,5]
d = {'fname': 'Ravi', 'lname': 'Y', 'hobbies': ['reading', 'hiking', 'biking']}
# Looping through list.
for i in l :
    print(i)

# Looping through dictionary
for i in d :
    print(d[i])



for i in d.keys() :
    print(i, d[i])


# In[60]:


# Print even numbers
for i in range(0,20,2) :
    print(i)

# Print odd numbers
for i in range(1,20,2) :
    print(i)

# Use range to generate indexes for a list
for i in range(len(l)) :
    print(l[i])


# In[61]:


# Create a list of 0 to 99 using range function.
l = list(range(10))
print(l)


# In[62]:



# Using list comprehension create a list e of even numbers from l
e = [i for i in l if i%2==0]
print(e)


# In[63]:


# Using list comprehension create a list f of tuples of number and its factors l
f = [(i, [j for j in range(2,i) if i%j == 0]) for i in range(10)]
print(f)


# In[64]:


# Using list comprehension create a dict f with key as integer upto 10, value as list of its factors
f = dict([(i, [j for j in range(2,i) if i%j == 0]) for i in range(10)])
print(f)


# In[71]:



# Short hand form of if
i = 10
m = True if i % 2 == 0 else False
print(m)


# In[70]:


# More list comprehension examples.
print([i for i in range(10) if i % 2 == 0])
print([(i, j) for i in range(10) for j in range(i)])
print([(i, j) for i in range(10) for j in range(i) if i%2 == 0 and j%2 == 0])

