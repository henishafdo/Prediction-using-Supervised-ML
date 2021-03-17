#!/usr/bin/env python
# coding: utf-8

# # sahaya henisha malar TSF task 1 mar 2021
# #predict percentage of a student based on the no.of study of hours
# #linear regression task
# 
# #import libraries and our data set
# import numpy as np 
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error

# # loading Datasets

# In[21]:


#read the data set
df = pd.read_csv("http://bit.ly/w-data")
print(df)


# In[13]:


#show the first five rows
df.head()


# In[4]:


#show the last five rows
df.tail()


# # Data preprocessing

# In[5]:


df.info()


# In[6]:


#view some basic details
df.describe()


# In[7]:


#rows and columns
df.shape


# In[16]:


#check if there any null value in the data set
df.isnull == True


# In[ ]:


# #There is no null value in the Dataset so,we can now visualize our data


# In[17]:


#plotting given data
plt.scatter(x=df.Hours, y=df.Scores)
plt.xlabel('student study Hours')
plt.ylabel('student percentage')
plt.title('scatter plot of student study hours vs student percentage')
plt.show()


# In[31]:


x = df.drop("Scores",axis="columns")
y = df.drop("Hours" ,axis="columns")
print("shape of x =", x.shape)
print("shape of y =", y.shape)


# In[46]:


x_train, x_test ,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=51)
print("shape of X_train =",x_train.shape)
print("shape of X_test =",x_test.shape)
print("shape of y_train =",y_train.shape)
print("shape of y_test =",y_test.shape)


# # select Model

# In[37]:


lr =LinearRegression()


# In[38]:


lr.fit(x_train,y_train)


# In[39]:


lr.coef_


# In[40]:


lr.intercept_


# In[41]:


lr.predict([[9.2]])[0][0].round(2)


# In[42]:


y_pred =lr.predict(x_test)
y_pred


# In[43]:


pd.DataFrame(np.c_[x_test,y_test,y_pred],columns = ["study_hours","student_percentage_original","student_percentage_pred"])


# In[48]:


lr.score(x_test, y_test)


# In[49]:


plt.scatter(x_train, y_train)


# In[51]:


plt.scatter(x_test,y_test)
plt.plot(x_train, lr.predict(x_train), color="g")


# In[54]:


from sklearn import metrics
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))


# In[60]:


hours =[9.25]
answer = lr.predict([hours])
print(answer)


# In[ ]:





# In[ ]:




