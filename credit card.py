#!/usr/bin/env python
# coding: utf-8

# # Problem Statement:
# 
# 
# 

# 
# ##               Credit Card Lead Prediction
# Happy Customer Bank is a mid-sized private bank that deals in all kinds of banking products, like Savings accounts, Current accounts, investment products, credit products, among other offerings.
# 
# The bank also cross-sells products to its existing customers and to do so they use different kinds of communication like tele-calling, e-mails, recommendations on net banking, mobile banking, etc. 
# 
# In this case, the Happy Customer Bank wants to cross sell its credit cards to its existing customers. The bank has identified a set of customers that are eligible for taking these credit cards.
# 
# Now, the bank is looking for your help in identifying customers that could show higher intent towards a recommended credit card, given:
# 
# Customer details (gender, age, region etc.)
# Details of his/her relationship with the bank (Channel_Code,Vintage, 'Avg_Asset_Value etc.)

#  # Table of Content
# Step 1: Importing the Relevant Libraries
# 
# Step 2: Data Inspection
# 
# Step 3: Data Cleaning
# 
# Step 4: Exploratory Data Analysis
# 
# Step 5: Building Model

# # Import libraries

# In[624]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# # Importing Data

# In[625]:


#Load, View and get a high level understanding of the given data to pandas dataframe
train= pd.read_csv('train_s3TEQDk.csv')
test=pd.read_csv('test_mSzZ8RL.csv')


# In[626]:


train.shape,test.shape


# __We have 245725 rows and 11 columns in Train set whereas Test set has 105312 rows and 10 columns.__

# In[627]:


#Ratio of null values
train.isnull().sum()/train.shape[0] *100


# In[628]:


#Ratio of null values
test.isnull().sum()/test.shape[0] *100


# * __We have 11%  missing values of Credit_Product column in both train and test data.__

# In[629]:


train.describe()


# ###### Observation:
# Atleast 75% of the customers was not interested for credit cards 
# 

# In[630]:


#categorical features
categorical = train.select_dtypes(include =[np.object])
print("Categorical Features in Train Set:",categorical.shape[1])

#numerical features
numerical= train.select_dtypes(include =[np.float64,np.int64])
print("Numerical Features in Train Set:",numerical.shape[1])


# In[631]:


#categorical features
categorical = test.select_dtypes(include =[np.object])
print("Categorical Features in Test Set:",categorical.shape[1])

#numerical features
numerical= test.select_dtypes(include =[np.float64,np.int64])
print("Numerical Features in Test Set:",numerical.shape[1])


# # Data Cleaning

# Why missing values treatment is required? Missing data in the training data set can reduce the power / fit of a model or can lead to a biased model because we have not analysed the behavior and relationship with other variables correctly. It can lead to wrong prediction.

# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


train.Is_Lead.describe()


# In[ ]:


#Imputing with Mode
train['Credit_Product']= train['Credit_Product'].fillna(train['Credit_Product'].mode()[0])
test['Credit_Product']= test['Credit_Product'].fillna(test['Credit_Product'].mode()[0])


# __Since the Credict_product is a categorical column, we can impute the missing values by "Mode"(Most Repeated Value) from the column__

# # Exploratory Data Analysis

# In[ ]:


train.columns


# In[ ]:


train.head(6)


# In[ ]:


train['Occupation'].value_counts()


# In[ ]:


plt.figure(figsize=(8,5))
sns.countplot('Occupation',data=train,palette='bright')


# According to occupation, most of customers are self employed

# In[ ]:


plt.figure(figsize=(8,5))
sns.countplot('Gender',data=train,palette='rocket')


# According to gender, most of customers are male 

# In[ ]:


plt.figure(figsize=(8,5))
sns.countplot(x='Is_Lead',hue="Occupation",data=train,palette='dark')


# * 70000 customers who are self employed was not interested for credit cards
# * 30000 customers who are self employed was intersted for credit cards

# In[ ]:


plt.figure(figsize=(8,5))
sns.countplot(x='Is_Lead',hue="Gender",data=train,palette='muted')


# 
# 
# *   1 lakh male customers were not interested for credit cards
# *   40000 male customers were interested for credit cards
# 
# 
# 
# 

# In[ ]:


plt.figure(figsize=(8,5))
sns.countplot(x='Is_Lead',hue="Credit_Product",data=train,palette='dark')


# 
# 
# *  1 lakh fourty thousand customers who are not bought any products was not interested for credit cards 
# *  Fourty thousand customers who are not bought any products was interested  for credit cards
# 
# 

# In[ ]:


plt.figure(figsize=(8,5))
sns.countplot(x='Is_Lead',hue="Is_Active",data=train,palette='dark')


# 
# 
# *  1 lakh twenty thousand customers who are not active was not interested for credit cards
# *  thirty thousand customers who are not active was interested for credit cards

# In[ ]:


plt.figure(figsize=(8,5))
sns.countplot(x='Gender',hue="Is_Active",data=train,palette='bright')


# 
# 
# *  Seventy thousand female customers who are not active was not interested for credit cards
# *  Eighty thousand female customers who are not active was interested for credit cards   
# 
# 

# In[ ]:


plt.figure(figsize=(8,5))
sns.countplot('Is_Lead',data=train,palette='ocean')


# 
# *  1 lakh Seventy five thousand customers was not interested for credit cards
# *  Sixty thousand customers  was interested for credit cards   
# 
# 

# In[ ]:


cat = ['Gender','Occupation','Credit_Product','Is_Active']
num = ['Age','Avg_Account_Balance','Vintage','Is_Lead']


# In[ ]:


train[cat].describe().T


# In[ ]:


train.Is_Lead.value_counts(normalize=True)


# In[ ]:


train[num].describe().T


# In[ ]:


corr = train[num].corr()
round(corr,2)


# In[ ]:


fig_dims = (17, 8)
fig = plt.subplots(figsize=fig_dims)
mask = np.triu(np.ones_like(corr, dtype=np.bool)) 
sns.heatmap(round(corr,2), annot=True, mask=mask)


# In[ ]:


# Labelencoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
var_mod = train.select_dtypes(include='object').columns
for i in var_mod:
    train[i] = le.fit_transform(train[i])
    
for i in var_mod:
    test[i] = le.fit_transform(test[i])


# * __Encoding the required columns from training and test dataset__

# In[ ]:


train.columns


# In[ ]:


columns=['Gender','Age', 'Occupation','Channel_Code','Region_Code',
       'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active',
       'Is_Lead']


# In[ ]:


data=train[columns]


# In[ ]:


# Splitting the dataset into the dependent variable and independent variable
X= data.drop(columns = ['Is_Lead'], axis=1)
y= data['Is_Lead']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[ ]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train= sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


classifier = Sequential()


# In[ ]:


# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu', input_dim =9))


# In[ ]:


# Adding the second hidden layer
classifier.add(Dense(units = 50, kernel_initializer = 'uniform', activation = 'relu'))


# In[ ]:


# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


# In[ ]:


# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[ ]:


classifier.fit(X_train, y_train, batch_size =100, epochs =10,verbose=1)


# In[ ]:


y_pred = classifier.predict(X_test)
np.round(y_pred)


# In[ ]:


tcolumns=['Gender','Age', 'Occupation','Channel_Code','Region_Code',
       'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active']


# In[ ]:


testdata=test[tcolumns]


# In[ ]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
testdata1= sc.fit_transform(testdata)


# In[ ]:


submission = pd.read_csv('sample_submission_eyYijxG.csv')
final_predictions = classifier.predict(testdata1)
submission['Is_Lead'] = np.round(final_predictions)
#only positive predictions for the target variable
submission['Is_Lead'] = submission['Is_Lead'].apply(lambda x: 0 if x<0 else x)
submission.to_csv('my_submission.csv', index=False)


# In[ ]:


from sklearn.metrics import roc_curve
y_pred = classifier.predict(X_test)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred)


# In[ ]:


from sklearn import metrics
auc=metrics.roc_auc_score(y_test, y_pred,average='macro', sample_weight=None, max_fpr=None, multi_class='raise', labels=None)


# In[ ]:


auc


# In[ ]:




