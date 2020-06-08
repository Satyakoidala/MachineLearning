#!/usr/bin/env python
# coding: utf-8

# # Jupyter Notebook Tips

# In[2]:


#Jupyter Notebook Tips
print?  #To get help on a function
print   #shift+tab after function name to get function signature
pwd # to get current working folder. If you are trying to open any files, path of file should be relative to pwd.


# In[4]:


def myPrint(inStr) :
    print(inStr)


# In[5]:


#Jupyter Notebook Tips
myPrint   # Tab anytime to get recommendations / auto complete 
myPrint?? # ?? to get definition of function.


# # String Formatting and Token Substitution

# In[24]:


i = 100
s = "I am a String"

print("Integer : %d, String : %s"%(i, s))
print("Integer : {}, String : {}".format(i, s))
print("Integer : {intParam}, String : {strParam}".format(intParam=i, strParam=s))
print("Integer : {intParam}, String : {strParam}".format(strParam=s, intParam=i))
print("Integer : {i}, String : {s}")
print(f"Integer : {i}, String : {s}")


# # Function parameters with default values

# In[8]:


# Function definition
def myAdd(a, b, c) :
    print(f"a : {a} b : {b} c : {c}")
    return a + b + c

# To call a function
print("1,2,3", myAdd(1,2,3))
# Error if called with less than 3 param
print("1,2", myAdd(1,2))


# In[17]:


# To make it behave with polymorphism, add default values to optional parameters
def myAdd(a=0, b=0, c=0) :
    print(f"a : {a} b : {b} c : {c}")
    return a + b + c

# Default value params should always be after regular parameters in function definition

# Works with no param
print("no param", myAdd())
# Works with one, two or three param
print("1 param", myAdd(1))
print("2 param", myAdd(1,2))
print("3 param", myAdd(1,2,3))


# # List as parameters

# In[12]:


l = [1,2,3]
print("list as param", myAdd(*l))

# all of below will work
l = [1,2,3]
print("list as param, 3 elements in list", myAdd(*l))
l = [1,2]
print("list as param, 2 elements in list", myAdd(*l))
l = [1]
print("list as param, 1 element in list", myAdd(*l))


# In[13]:


# Gives Error on below, as list has more than three param
l = [1,2,3,4]
print("list as param", myAdd(*l))


# In[14]:


# You can call function with name value parameters
print("name value params", myAdd(c=3))


# # Dictionary values as function parameters

# In[25]:


d = {'c':3}
print("dict as param with c", myAdd(**d))
d = {'c':3, 'a':1}
print("dict as param with c,a", myAdd(**d))
d = {'c':3, 'a':1, 'b':2}
print("dict as param with c,a,b", myAdd(**d))


# # Defining functions that can take variable number of parameters

# In[26]:


def myAdd(a=0, b=0, c=0, *l) :
    print(f"a : {a} b : {b} c : {c}")
    total = a + b + c
    print("l", type(l), l)
    for i in l :
        if type(i).__name__ == 'int' :
            total += i
    return total


# In[27]:


myAdd(1,2,3,4,5)


# In[28]:


myAdd(1,2,3,4,5,6,7,8,9)


# In[30]:


l = list(range(100))
myAdd(*l)


# In[31]:


myAdd(a=1)


# In[32]:


myAdd(a=1, d=3) # Gives error as d is not defined as parameter to myAdd


# In[33]:


def myAdd(a=0, b=0, c=0, *l, **d) :
    print(f"a : {a} b : {b} c : {c}")
    total = a + b + c
    print("l", type(l), l)
    
    count=3
    for i in l :
        if type(i).__name__ == 'int' :
            total += i
            count+=1

    print("d", type(d), d)
    for i in d :
        if type(d[i]).__name__ == 'int' :
            total += d[i]
            count+=1

    return total, count
# Function can return multiple parameters


# In[34]:


myAdd(a=1, d=4)


# In[39]:


myAdd(1,2,3,4,5,d=10,e=20)


# # OOP - Classes / Objects

# In[61]:


class Bag :
    width=10
    height=9
    contents = []
    
    def __init__(self, inputType="None") :
        self.type = inputType
        
    def getType(self) :
        return self.type
    
    def addItem(self, item) :
        self.contents.append(item)
    
    def getContents(self) :
        return tuple(self.contents)
    
    def __add__(self, bag) :
        return self.getContents() + bag.getContents()


# In[63]:


b = Bag("Bag")
b.getType()


# In[47]:



b = Bag('small')
b.getType()


# In[84]:


class LaptopBag(Bag) :
    def __init__(self, inputSize = 14) :
        super().__init__("Laptop")
        #Bag.__init__(self, 'Laptop')
        self.laptopSize = inputSize

    def getLaptopSize(self) :
        return self.laptopSize
    
l = LaptopBag(20)
l.getType()
l.getLaptopSize()


# In[83]:


class SchoolBag(Bag) :
    def __init__(self, inputSize = "small") :
        super().__init__("School")
        #Bag.__init__(self, 'School')
        self.schoolBagSize = inputSize

    def getSchoolBagSize(self) :
        return self.schoolBagSize
    
s = SchoolBag("large")
print(s.getType())
print(s.getSchoolBagSize())


# In[81]:


class Bag :
    width=10
    height=9
    
    def __init__(self, inputType="None") :
        self.type = inputType
        self.contents = []
        
    def getType(self) :
        return self.type
    
    def addItem(self, item) :
        self.contents.append(item)
    
    def getContents(self) :
        return tuple(self.contents)
    
    def __add__(self, bag) :
        return self.getContents() + bag.getContents()


# In[68]:


b1 = Bag("Bag1")
b1.addItem("item1")
print(b1.getContents())


# In[69]:


b2 = Bag("Bag2")
b2.addItem("item2")
print(b2.getContents())


# In[70]:


b1 + b2


# In[85]:


class laptopSchoolBag(LaptopBag, SchoolBag) :
    def __init__(self, laptopSize= 15, bagSize="Medium") :
        LaptopBag.__init__(self, laptopSize)
        SchoolBag.__init__(self, bagSize)
        Bag.__init__(self, "**Laptop School Bag**")
        


# In[89]:


lsb = laptopSchoolBag()
print(lsb.getType())
print(lsb.getSchoolBagSize())
print(lsb.getLaptopSize())
print(lsb.getContents())
lsb.addItem("Pen")
print(lsb.getContents())
lsb.addItem("Book")
lsb.addItem("Laptop")
print(lsb.getContents())


# In[92]:


# __name__ == "__main__" when module executed directly instead of import, __name__ will have value as "__main__"
# Function is a object also, you can pass function as parameter or return value
# import <module name> without .py will execute contents of .py in current namespace
# Import can done in below two ways.
### import sys
### from sys import argv
# Same as above with alias
### import sys as s
### from sys import argv as arg

# global to update module variables inside functions
# classes are defined in their own .py files and imported into other modules.


# In[95]:


# Exception block does not print
try :
    a = int('10')
    print("After exception")
except :
    print("In exception block")
finally:
    print("prints always")
    


# In[96]:


# Exception block prints, but print after int function does not print
try :
    a = int('a')
    print("After exception")
except :
    print("In exception block")
finally:
    print("prints always")


# In[97]:


#To get information about exception  
try :
    a = int('a')
    print("After exception")
except Exception as e:
    print("In exception block", e)
finally:
    print("prints always")


# In[98]:


pwd


# In[ ]:




