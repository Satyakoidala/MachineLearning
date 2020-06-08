#!/usr/bin/env python
# coding: utf-8

# In[41]:


from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[52]:


iris_dataset = load_iris()


# In[49]:


print("Keys of iris_dataset:\n", iris_dataset.keys())
print( "feature names", iris_dataset['feature names'])
print( type(iris_dataset), iris_dataset.data[:10])


# In[50]:


print("Keys of iris_dataset:\n", iris_dataset.keys())

# ['data', 'target', 'target_names', 'DESCR', 'feature_names'])

print(iris_dataset['DESCR'][:193] + "\n...")
print("Target names:", iris_dataset['target_names'])
print("Feature names:\n", iris_dataset['feature_names'])
print("Target names:", iris_dataset['target_names'])
print("Feature names:\n", iris_dataset['feature_names'])
print("Type of data:", type(iris_dataset['data']))
print("Shape of data:", iris_dataset['data'].shape)
print("First five rows of data:\n", iris_dataset['data'][:5])
print("Type of target:", type(iris_dataset['target']))
print("Shape of target:", iris_dataset['target'].shape)
print("Target:\n", iris_dataset['target'])


# In[53]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)


# In[54]:


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# In[55]:


iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20})
plt.show()


# In[7]:


from sklearn import metrics #for checking the model accuracy


# In[56]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


# In[57]:


# Predict for test dataset set aside
y_pred = knn.predict(X_test)
print("Test set predictions:\n", y_pred)
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
print("Test set score using Metrics : ", metrics.accuracy_score(y_pred,y_test))


# In[10]:


# Predict for a new data element
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape:", X_new.shape)
prediction = knn.predict(X_new)
print("Prediction:", prediction)
print("Predicted target name:", iris_dataset['target_names'][prediction])


# In[11]:


# Decison Tree Classifier
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)


# In[12]:


print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
print("Test set score using Metrics : ", metrics.accuracy_score(y_pred,y_test))


# In[13]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm

logr = LogisticRegression()
logr.fit(X_train,y_train)


# In[14]:


y_pred = logr.predict(X_test)
acc_log = metrics.accuracy_score(y_pred,y_test)
print('The accuracy of the Logistic Regression is', acc_log)


# In[15]:


# Logistic Regression
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm

sv = svm.SVC() #select the algorithm
sv.fit(X_train,y_train) # we train the algorithm with the training data and the training output


# In[16]:


y_pred = sv.predict(X_test) #now we pass the testing data to the trained algorithm
acc_svm = metrics.accuracy_score(y_pred,y_test)
print('The accuracy of the SVM is:', acc_svm)


# In[17]:


from sklearn.linear_model import LinearRegression
X = np.array([[2],[3],[4],[5]])
y = np.array([[4],[6],[8],[10]])


# In[18]:


lr = LinearRegression().fit(X,y)


# In[19]:


print("w[0]: %f  b: %f" % (lr.coef_[0], lr.intercept_))


# In[20]:


plt.figure(figsize=(8, 8))
X_test = np.array([[100],[300],[400],[500]])
plt.plot(X_test, lr.predict(X_test))


# In[21]:


from sklearn.datasets import load_boston
boston = load_boston()


# In[22]:


print(boston.keys())
print(boston.data.shape)
print(boston.feature_names)
print(boston.DESCR)


# In[23]:


bos = pd.DataFrame(boston.data, columns=boston.feature_names)
print(bos.head())


# In[24]:


print(boston.target.shape)

bos['PRICE'] = boston.target
print(bos.head())

print(bos.describe())

X = bos.drop('PRICE', axis = 1)
Y = bos['PRICE']


# In[25]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[26]:


from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, Y_train)


# In[29]:


from sklearn.metrics import mean_squared_error
Y_pred = lm.predict(X_test)

plt.scatter(Y_test, Y_pred)
plt.xlabel("Prices: ")
plt.ylabel("Predicted prices: ")
plt.show()

#mse = mean_squared_error(Y_test, Y_pred)
#print(mse)
print("Linear Train Score : ", lm.score(X_train, Y_train))
print("Linear Test Score : ", lm.score(X_test, Y_test))


# In[30]:


from sklearn.linear_model import Ridge
# L2 Regularization

ridge1 = Ridge().fit(X_train, Y_train)
#Y_pred = ridge.predict(X_test)

#ridge_mse = mean_squared_error(Y_test, Y_pred)
#print(ridge_mse)
print("Ridge Train Score : ", ridge1.score(X_train, Y_train))
print("Ridge Test Score : ", ridge1.score(X_test, Y_test))


# In[31]:


ridge10 = Ridge(alpha=10).fit(X_train, Y_train)
#Y_pred = ridge10.predict(X_test)

print("Ridge Train Score : ", ridge10.score(X_train, Y_train))
print("Ridge Test Score : ", ridge10.score(X_test, Y_test))


# In[32]:


ridge01 = Ridge(alpha=0.1).fit(X_train, Y_train)
#Y_pred = ridge01.predict(X_test)

print("Ridge Train Score : ", ridge01.score(X_train, Y_train))
print("Ridge Test Score : ", ridge01.score(X_test, Y_test))


# In[33]:


ridge001 = Ridge(alpha=0.01).fit(X_train, Y_train)
#Y_pred = ridge001.predict(X_test)

print("Ridge Train Score : ", ridge001.score(X_train, Y_train))
print("Ridge Test Score : ", ridge001.score(X_test, Y_test))


# In[34]:


# Lesso

from sklearn.linear_model import Lasso
# L1 Regularization

lasso1 = Lasso().fit(X_train, Y_train)

print("Lasso Train Score : ", lasso1.score(X_train, Y_train))
print("Lasso Test Score : ", lasso1.score(X_test, Y_test))
print("Number of features used : ", np.sum(lasso1.coef_ != 0))


# In[35]:


lasso1 = Lasso(max_iter=10000).fit(X_train, Y_train)

print("Lasso Train Score : ", lasso1.score(X_train, Y_train))
print("Lasso Test Score : ", lasso1.score(X_test, Y_test))
print("Number of features used : ", np.sum(lasso1.coef_ != 0))


# In[36]:


lasso01 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, Y_train)

print("Lasso Train Score : ", lasso01.score(X_train, Y_train))
print("Lasso Test Score : ", lasso01.score(X_test, Y_test))
print("Number of features used : ", np.sum(lasso01.coef_ != 0))


# In[37]:


lasso001 = Lasso(alpha=0.001, max_iter=100000).fit(X_train, Y_train)

print("Lasso Train Score : ", lasso001.score(X_train, Y_train))
print("Lasso Test Score : ", lasso001.score(X_test, Y_test))
print("Number of features used : ", np.sum(lasso001.coef_ != 0))


# In[38]:


iris_dataset.feature_names


# In[39]:


from sklearn.cluster import KMeans

X = iris_dataset.data   #[:, [0,2]]
y = iris_dataset.target

km = KMeans(n_clusters = 3, random_state=21) # , n_jobs = 4
km.fit(X)

# Dataset Slicing
x_axis = iris_dataset.data[:, 0]  # Sepal Length
y_axis = iris_dataset.data[:, 2]  # petal Length

plt.subplot(1,2,1)

plt.scatter(x_axis, y_axis, c=km.labels_)

plt.subplot(1,2,2)
plt.scatter(x_axis, y_axis, c=iris_dataset.target)
plt.show()


# In[40]:


from sklearn.cluster import KMeans

X = iris_dataset.data   #[:, [0,2]]
y = iris_dataset.target

km = KMeans(n_clusters = 3, random_state=21) # , n_jobs = 4
km.fit(X)

# Dataset Slicing
x_axis = iris_dataset.data[:, 1]  # Sepal Width
y_axis = iris_dataset.data[:, 3]  # petal Width

plt.subplot(1,2,1)

plt.scatter(x_axis, y_axis, c=km.labels_)

plt.subplot(1,2,2)
plt.scatter(x_axis, y_axis, c=iris_dataset.target)
plt.show()


# In[ ]:




