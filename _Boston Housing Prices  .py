#!/usr/bin/env python
# coding: utf-8

# # Introduction

#  Welcome to the Boston Housing Price Prediction Project! Our mission is to forecast the median value of owner-occupied homes in Boston suburbs using regression algorithms like Linear Regression and Random Forest. This endeavor involves honing skills in feature scaling and cross-validation for robust model performance. Leveraging the Boston Housing dataset, we'll delve into feature engineering, optimizing our model's predictive capabilities. The assessment will be anchored in the Mean Squared Error metric, ensuring accurate predictions. This project not only seeks to provide valuable insights into housing prices but also aims to deliver a practical tool for real estate professionals, enhancing decision-making in Boston's dynamic real estate market. Join us on this data-driven journey to transform housing market predictions!

# # Import libraries necessary for this project

# In[1]:


import numpy as np
import pandas as pd
import os
import seaborn as sns 
import matplotlib.pyplot as plp 
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings 
warnings.filterwarnings('ignore')


# # Loading the dataset

# In[2]:


#os.chdir
os.chdir("E:/Study/Projects/Machine learning/3rd Boston Housing Prices")
current_directory = os.getcwd()
files = os.listdir(current_directory)


# In[3]:


dB=pd.read_csv("Boston Dataset.csv")
dB.drop(columns=['Unnamed: 0'],axis=0, inplace = True)
dB.head()


# In[4]:


(dB.shape)


# In[5]:


dB.describe()


# In[6]:


#data type informations 

dB.info()


# # Data Preprocessing   
# Handle Missing Data  

# In[7]:


# check for null values
dB.isnull().sum()


# # Exploratory Data Analysis

# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df is your DataFrame with multiple features
# If your DataFrame contains non-numeric columns, you might want to select only numeric columns
numeric_features = dB.select_dtypes(include=['float64', 'int64'])

# Set the size of the plot
plt.figure(figsize=(16, 8))

# Create a box plot for each numeric feature
sns.boxplot(data=numeric_features)

# Customize the plot
plt.title('Box Plots for Multiple Features')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

# Show the plot
plt.show()


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is your DataFrame with multiple features
# If your DataFrame contains non-numeric columns, you might want to select only numeric columns
numeric_features = dB.select_dtypes(include=['float64', 'int64'])

# Set up a 2x7 grid for 14 features
fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(16, 8))
fig.suptitle('Individual Box Plots for Multiple Features')

# Flatten the 2D array of subplots
axes = axes.flatten()

# Create box plots for each numeric feature
for i, feature in enumerate(numeric_features.columns):
    sns.boxplot(x=numeric_features[feature], ax=axes[i])
    axes[i].set_title(feature)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Show the plot
plt.show()


# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is your DataFrame with multiple features
# If your DataFrame contains non-numeric columns, you might want to select only numeric columns
numeric_features = dB.select_dtypes(include=['float64', 'int64'])

# Set up a 2x7 grid for 14 features
fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(16, 8))
fig.suptitle('Individual Distribution Plots for Multiple Features')

# Flatten the 2D array of subplots
axes = axes.flatten()

# Create distribution plots for each numeric feature
for i, feature in enumerate(numeric_features.columns):
    sns.histplot(numeric_features[feature], kde=True, ax=axes[i], bins=20)
    axes[i].set_title(feature)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Show the plot
plt.show()


# ## Min-Max Normalization

# In[11]:


cols = ['crim','indus','tax','lstat']
for col in cols:
    # find minimum and maximum of that column
    minimum = min(dB[col])
    maximum = max(dB[col])
    dB[col] = (dB[col] - minimum) / (maximum - minimum)


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is your DataFrame with multiple features
# If your DataFrame contains non-numeric columns, you might want to select only numeric columns
numeric_features = dB.select_dtypes(include=['float64', 'int64'])

# Set up a 2x7 grid for 14 features
fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(16, 8))
fig.suptitle('Individual Distribution Plots for Multiple Features')

# Flatten the 2D array of subplots
axes = axes.flatten()

# Create distribution plots for each numeric feature
for i, feature in enumerate(numeric_features.columns):
    sns.histplot(numeric_features[feature], kde=True, ax=axes[i], bins=20)
    axes[i].set_title(feature)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Show the plot
plt.show()


# In[13]:


# standardization
from sklearn import preprocessing
scalar = preprocessing.StandardScaler()

# fit our data 
scaled_cols = scalar.fit_transform(dB[cols])
scaled_cols = pd.DataFrame(scaled_cols, columns=cols)
scaled_cols.head()


# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is your DataFrame with multiple features
# If your DataFrame contains non-numeric columns, you might want to select only numeric columns
numeric_features = dB.select_dtypes(include=['float64', 'int64'])

# Set up a 2x7 grid for 14 features
fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(16, 8))
fig.suptitle('Individual Distribution Plots for Multiple Features')

# Flatten the 2D array of subplots
axes = axes.flatten()

# Create distribution plots for each numeric feature
for i, feature in enumerate(numeric_features.columns):
    sns.histplot(numeric_features[feature], kde=True, ax=axes[i], bins=20)
    axes[i].set_title(feature)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Show the plot
plt.show()


# # coorelation matrix

# In[15]:


corr = dB.corr()
plt.figure(figsize = (15,10))
sns.heatmap(corr, annot=True , cmap='coolwarm' )


# In[16]:


sns.regplot(y=dB['medv'],x=dB['lstat'])


# In[17]:


sns.regplot(y=dB['medv'], x=dB['rad'])


# # Data preparation 

# In[18]:


y = dB["medv"]
y


# In[19]:


x = dB.drop("medv" ,axis=1)
x


# # Train - Test splits

# In[20]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2 , random_state=100)


# In[21]:


x_train


# In[22]:


x_test


# # Model building 

# # LinearRegression 

# In[29]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the training set
model.fit(x_train, y_train)

#Make prediction on the test set 
predictions = model.predict(x_test)

#Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error : {mse}')


# In[34]:


from sklearn.model_selection import cross_val_score

# Perform cross-validated scoring
# cv parameter specifies the number of folds in cross-validation
# scoring parameter specifies the evaluation metric (default is R^2 score)
cv_scores = cross_val_score(model, x, y, cv=5)  # 5-fold cross-validation

# Calculate and print the mean of the cross-validated scores
mean_cv_score = cv_scores.mean()
print("Mean Cross-validated Score:", mean_cv_score)


# # Random Forest 

# In[38]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Initialize the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100 ,random_state=42)

#Train the model on the training set
rf_model.fit(x_train,  y_train)

#Make predictions on the test set
rf_predictions = rf_model.predict(x_test)

#Evaluate the model
rf_mse = mean_squared_error(y_test, rf_predictions)
print(f'Random forest Mean Squared Error:{rf_mse}')


# In[41]:


from sklearn.model_selection import cross_val_score

# Perform cross-validated scoring
# cv parameter specifies the number of folds in cross-validation
# scoring parameter specifies the evaluation metric (default is R^2 score)
cv_scores_rf = cross_val_score(rf_model, x, y, cv=5)  # 5-fold cross-validation

# Display the cross-validated scores
print("Random Forest Cross-validated Scores:", cv_scores_rf)

