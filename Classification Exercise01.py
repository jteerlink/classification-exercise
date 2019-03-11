#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


train = pd.read_csv('data/exercise_01_train.csv')
test = pd.read_csv('data/exercise_01_test.csv')


# In[187]:


train.isnull().sum(axis=1).sum()

# for x in row_null_count[row_null_count>=1]:
#     print(x)

# for i in row_null_count:
#     print(i)


# In[5]:


type(train.x45[0])


# In[6]:


train.x45 = train.x45.str.strip('%')
train.x45 = train.x45.astype(float)
train.x45.isnull().sum()


        


# In[7]:


train.x41= train.x41.str.strip('$').astype(float)
type(train.x45[0])


# In[8]:


x34_values = set(train.x34)
for i in x34_values:
    print(i)
    
# train[train.x34.isnull()]

train['x34'].replace(to_replace='mercades',value='mercedes',inplace=True)
train.x34 = train['x34'].str.lower()


# In[9]:


brand_count = pd.DataFrame(train.x34.groupby(train.x34).count())
brand_count


# In[10]:


brand_count.plot.bar()


# In[11]:


x35_values = set(train.x35)
for i in x35_values:
    print(i)
    
train[train.x35.isnull()]

train.x35.replace('wed','wednesday',inplace=True)
train.x35.replace('fri','friday',inplace=True)
train.x35.replace('thur','thursday',inplace=True)
train.x35.replace('thurday','thursday',inplace=True)


# In[12]:


day_count = pd.DataFrame(train.x35.groupby(train.x35).count())
day_count.plot.bar()

# plt.bar(x35_values,train.x35)


# In[13]:


x68_values = set(train.x68)
for i in x68_values:
    print(i)
    
# train[train.x68.isnull()]

train.x68 = train.x68.str.lower()


# In[14]:


def clean(original,new):
    train.x68.replace(original,new,inplace=True)

dict = { 'sept.':'sep','july':'jul','january':'jan','dev':'dec'}


for key,value in dict.items():
    clean(key,value)
    
    


# In[15]:


month_count = pd.DataFrame(train.x68.groupby(train.x68).count())
month_count.plot.bar()


# In[16]:


x93_values = set(train.x93)
for i in x93_values:
    print(i)
    
train[train.x93.isnull()]


# In[17]:


country_count = pd.DataFrame(train.x93.groupby(train.x93).count())
country_count.plot.bar()


# In[18]:


train.isnull().sum(axis=1).sum()/train.y.count()

# records with missing values ~2%, too many to just delete


# In[19]:


train.y.sum()/train.y.count()

#positive label ratio >5%, would not consider minority class


# In[72]:


# # fill categorical with most common values


fill_cat_var = {
'x34':'volkswagon',
'x35':'wednesday',
'x68':'july',
'x93':'asia'}


for key,value in fill_cat_var.items():
    train[key].fillna(value,inplace=True)
    
    


# # median imputation and setting up dfs for models

# In[ ]:


train.fillna(value=np.nan)
train_numeric = train.drop(labels = ['x34','x35','x68','x93'],axis=1)


# In[59]:


from sklearn.impute import SimpleImputer
imp_median = SimpleImputer(missing_values=np.nan, strategy='median')

train_numeric_imp = pd.DataFrame(imp_median.fit_transform(train_numeric))


# In[63]:


col_old = train_numeric_imp.columns.values.tolist()
cols = train_numeric.columns.tolist()

col_names = {i: j for i, j in zip(col_old,cols)}

train_numeric_imp = train_numeric_imp.rename(mapper=col_names,axis=1)


# In[ ]:


train_numeric_imp['x34'] = train['x34']
train_numeric_imp['x35'] = train['x35']
train_numeric_imp['x68'] = train['x68']
train_numeric_imp['x93'] = train['x93']

train_X = train_numeric_imp.drop(labels = ['y'],axis=1)
train_y = train['y']


# In[ ]:


for col in train_X.columns:
    if(train_X[col].dtype == 'object'):
        train_X[col]= train_X[col].astype('category')
        train_X[col] = train_X[col].cat.codes


# In[ ]:





# ## Model 1: Logistic Regression

# In[135]:


# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(train_X)  
# train_X_scaled = scaler.transform(train_X)
# # X_test = scaler.transform(X_test)


# In[141]:


from sklearn.linear_model import LogisticRegressionCV

logit = LogisticRegressionCV(cv=5,n_jobs=-1,max_iter=20000,refit=True).fit(train_X,train_y)

#


# In[142]:


predict = logit.predict_proba(train_X)

logit.score(train_X,train_y)


# In[143]:


predict_binary = logit.predict(train_X)
predict_binary


# In[144]:


from sklearn.metrics import roc_auc_score
roc_auc_score(train_y, predict_binary)


# In[155]:


average_precision_score(train_y, predict_binary)


# # Model 2: RandomForest Classifier

# In[99]:


from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [500,600,700,800]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 50, num = 5)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf
               }


# In[100]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 20, cv = 3, verbose=2, random_state=1, n_jobs = -1)
# Fit the random search model
rf_random.fit(train_X,train_y)


# In[103]:


rf_random_best = rf_random.best_params_

rf_random_best


# In[ ]:


{'n_estimators': 800,
 'min_samples_split': 5,
 'min_samples_leaf': 1,
 'max_features': 'auto',
 'max_depth': 50}


# In[106]:


rfcv = RandomForestClassifier(n_estimators=800,min_samples_split=5,min_samples_leaf=1,max_features='auto',max_depth=50)

rfcv.fit(train_X,train_y)


# In[128]:


predict = rfcv.predict(train_X)


# In[131]:


# print(predict)
predictdf = pd.DataFrame(predict)

# predictdf['0']
resultsdf = pd.DataFrame({'Train Label':train_y,'Predicted Label':predictdf[0]})


# In[132]:


from sklearn.metrics import roc_auc_score
auc = roc_auc_score(resultsdf['Train Label'],resultsdf['Predicted Label'])

from sklearn.metrics import average_precision_score
precision  = average_precision_score(resultsdf['Train Label'],resultsdf['Predicted Label'])

print(auc,precision)


# In[133]:


rfcv.score(train_X,predictdf[0])


# ### Test values
# 

# In[188]:


#convert string currency to float
test.x41= test.x41.str.strip('$').astype(float)
type(test.x45[0])

test.x45 = test.x45.str.strip('%')
test.x45 = test.x45.astype(float)
test.x45.isnull().sum()

#correct mispellings
test['x34'].replace(to_replace='mercades',value='mercedes',inplace=True)
test.x34 = test['x34'].str.lower()


#brute force method
test[test.x35.isnull()]

test.x35.replace('wed','wednesday',inplace=True)
test.x35.replace('fri','friday',inplace=True)
test.x35.replace('thur','thursday',inplace=True)
test.x35.replace('thurday','thursday',inplace=True)

#UDF method -- much easier
def clean(original,new):
    test.x68.replace(original,new,inplace=True)

dict = { 'sept.':'sep','july':'jul','january':'jan','dev':'dec'}

for key,value in dict.items():
    clean(key,value)
    


# In[190]:


#ensure missing values are properly encoded as NaN
test.fillna(value=np.nan)
test_numeric = test.drop(labels = ['x34','x35','x68','x93'],axis=1)

#impute missing values with median, ranges on columns are not very large and have dont have much variance

from sklearn.impute import SimpleImputer
imp_median = SimpleImputer(missing_values=np.nan, strategy='median')

test_numeric_imp = pd.DataFrame(imp_median.fit_transform(test_numeric))



col_old = test_numeric_imp.columns.values.tolist()
col_new = test_numeric.columns.tolist()

#list/dictionary complication
col_names = {i: j for i, j in zip(col_old,col_new)}

test_numeric_imp = test_numeric_imp.rename(mapper=col_names,axis=1)


#readding categorical variables to imputed df
test_numeric_imp['x34'] = test['x34']
test_numeric_imp['x35'] = test['x35']
test_numeric_imp['x68'] = test['x68']
test_numeric_imp['x93'] = test['x93']

test_X = test_numeric_imp

#filled with mode 
for col in test_X.columns:
    if(test_X[col].dtype == 'object'):
        test_X[col]= test_X[col].fillna(test_X[col].mode()[0])



#code categorical variables to numeric to allow algorithms to include these variables
for col in test_X.columns:
    if(test_X[col].dtype == 'object'):
        test_X[col]= test_X[col].astype('category')
        test_X[col] = test_X[col].cat.codes


# In[198]:


test_logit = pd.DataFrame(logit.predict_proba(test_X))

test_logit[1].to_csv('results/results1.csv',index=False,sep=',',header='y')


# In[199]:


test_rfclf = pd.DataFrame(rfcv.predict_proba(test_X))

test_rfclf[1].to_csv('results/results2.csv',index=False,sep=',',header='y')

