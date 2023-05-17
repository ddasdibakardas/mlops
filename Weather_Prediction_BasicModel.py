#!/usr/bin/env python
# coding: utf-8

# # Regression on Weather Data 
# ### Target: Temperature- [Dataset](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data)
# ### Deployed Version- [Website](https://ddasdibakardas-mlops-app-3h5np3.streamlit.app/)

# In[1]:


# Import necessary libraries for data analysis, visualization, and machine learning

import numpy as np # NumPy for numerical computing and array operations
import pandas as pd # Pandas for data manipulation and analysis
import math # Python math library for mathematical operations

import matplotlib.pyplot as plt # Matplotlib for data visualization
import plotly.express as px # Plotly for interactive data visualization
import seaborn as sns # Seaborn for statistical data visualization

from sklearn.metrics import r2_score # Scikit-learn for R-squared metric
from sklearn.metrics import mean_squared_error # Scikit-learn for mean squared error metric
from sklearn.metrics import mean_absolute_error # Scikit-learn for mean absolute error metric

#pip install keras
from sklearn import linear_model # Scikit-learn for linear regression and other linear models
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor 
import tensorflow as tf
from tensorflow import keras
from sklearn import tree


import warnings
warnings.filterwarnings("ignore") # Disable warning messages for cleaner output


# In[2]:


# Read in two CSV files as Pandas DataFrame objects

data_train = pd.read_csv(r'DailyDelhiClimateTrain.csv') # Training data file
data_test  = pd.read_csv(r'DailyDelhiClimateTest.csv' ) # Test data file


# ## EDA

# In[3]:


data_train.head()


# In[4]:


data_test.head()


# In[5]:


data_train.describe()


# In[6]:


# Calculate the correlation matrix and generate a heatmap

corr = data_train.corr() # Calculate the pairwise correlations between columns

plt.figure(figsize=(20,5)) # Set the figure size for the heatmap
sns.heatmap(data_train.corr(),annot=True,annot_kws={'size':11},cmap='coolwarm') # Create the heatmap with annotations
plt.show() # Show the plot


# ### Outlier Removal 
# The pressure correlation plot shows that there is very little correlation so removing them can increase the models accuracy

# In[7]:


# Generate a box plot to visualize the distribution of the 'meanpressure' column for train
fig = px.box(data_train, y='meanpressure')
fig.show()


# In[8]:


data_train = data_train[(data_train['meanpressure'] > 950) & (data_train['meanpressure'] < 1200)] #droping the outliers


# In[9]:


# Generate a box plot to visualize the distribution of the 'meanpressure' column after removing the outliers
fig = px.box(data_train, y='meanpressure')
fig.show()


# In[10]:


# Generate a box plot to visualize the distribution of the 'meanpressure' column
fig = px.box(data_test, y='meanpressure')
fig.show()


# In[11]:


data_test = data_test[(data_test['meanpressure'] > 1000)]  #droping the outliers


# In[12]:


# Generate a box plot to visualize the distribution of the 'meanpressure' column
fig = px.box(data_train, y='meanpressure')
fig.show()


# In[13]:


# Calculate the correlation matrix and generate a heatmap

corr = data_train.corr() # Calculate the pairwise correlations between columns

plt.figure(figsize=(20,5)) # Set the figure size for the heatmap
sns.heatmap(data_train.corr(),annot=True,annot_kws={'size':11},cmap='coolwarm') # Create the heatmap with annotations
plt.show() # Show the plot


# In[14]:


# Calculate the correlation matrix and generate a heatmap

corr = data_test.corr() # Calculate the pairwise correlations between columns

plt.figure(figsize=(20,5)) # Set the figure size for the heatmap
sns.heatmap(data_test.corr(),annot=True,annot_kws={'size':11},cmap='coolwarm') # Create the heatmap with annotations
plt.show() # Show the plot


# ## Dataframe Visualisation

# In[15]:


# Generate a subplot of time series plots for each column in the 'data_train' DataFrame for training data

plt.figure(figsize=(26, 18)) # Set the figure size for the subplot
colors = ['','#fa6616', '#6c9497', '#5a00b3', '#899ad6'] # Set the colors for each subplot

for i, j in enumerate(data_train.columns): # Iterate over each column in the DataFrame
    if i>0: # Skip the first column (Date)
        plt.subplot(len(data_train.columns) + 1, 1, i + 1) # Create a new subplot
        plt.plot(data_train[j], color=colors[i]); # Plot the data for the current column
        plt.plot(data_train[j].rolling(25).mean()); # Plot the rolling average line for the current column
        plt.ylabel(j, fontsize=16) # Add the column name as the y-axis label
        plt.grid() # Add a grid to the plot
        plt.xlabel('Date', fontsize=16) # Add 'Date' as the x-axis label


# In[16]:


sns.pairplot(data_train)


# In[17]:


# Generate a subplot of time series plots for each column in the 'data_train' DataFrame

plt.figure(figsize=(26, 18)) # Set the figure size for the subplot
colors = ['','#fa6616', '#6c9497', '#5a00b3', '#899ad6'] # Set the colors for each subplot

for i, j in enumerate(data_train.columns): # Iterate over each column in the DataFrame
    if i>0: # Skip the first column (Date)
        plt.subplot(len(data_test.columns) - 1, 1, i) # Create a new subplot
        plt.plot(data_test[j], color=colors[i]); # Plot the data for the current column
        plt.plot(data_test[j].rolling(5).mean()); # Plot the rolling average line for the current column
        plt.ylabel(j, fontsize=16) # Add the column name as the y-axis label
        plt.grid() # Add a grid to the plot
        plt.xlabel('Date', fontsize=16) # Add 'Date' as the x-axis label


# In[18]:


sns.pairplot(data_test)


# ## Data Preprocessing

# In[19]:


X_train=data_train[['humidity','wind_speed','meanpressure']] # Assign the features for the training data set
Y_train=data_train[['meantemp']] # Assign the target variable for the training data set
X_test=data_test[['humidity','wind_speed','meanpressure']] # Assign the features for the testing data set
Y_test=data_test[['meantemp']] # Assign the target variable for the testing data set


# ## Linear Regression Model
# 
# ![image.png](attachment:image.png)

# In[20]:


linear_regressor = linear_model.LinearRegression() # Create a linear regression object
linear_regressor.fit(X_train, Y_train) # Fit the model to the training data set


# In[21]:


print(linear_regressor.coef_) # Print the coefficients of the model


# In[22]:


Y_pred=linear_regressor.predict(X_test) # Use the trained model to make predictions on the test data set
Y_pred # Print the predicted values


# In[23]:


Y_pred=Y_pred.flatten() # Flatten the predicted values array
Y_pred=np.absolute(Y_pred) # Take the absolute values of the predicted values
Y_pred=pd.DataFrame(Y_pred) # Convert the predicted values to a Pandas DataFrame


# In[24]:


Y_pred['Actual']=Y_test.to_numpy() # Add the actual 'meantemp' values from the test data set to the DataFrame
Y_pred.rename(columns={0:'Predicted'},inplace=True) # Rename the column containing the predicted values


# In[25]:


Y_pred #viewing the dataset


# In[26]:


accuracy=r2_score(Y_pred['Actual'],Y_pred['Predicted'])*100 # Calculate the accuracy of the linear regression model
print(" Accuracy of the model is %.3f" %abs(accuracy)) # Print the accuracy of the model as a percentage


# In[27]:


# Calculate root mean squared error of predicted and actual values
score = np.sqrt(mean_squared_error(Y_pred['Actual'],Y_pred['Predicted']))

# Print the result using formatted string
print("The Mean Squared Error of our Model is {}".format(round(score, 3)))


# In[28]:


# Calculate mean absolute error of predicted and actual values
score = np.sqrt(mean_absolute_error(Y_pred['Actual'],Y_pred['Predicted']))

# Print the result using formatted string
print("The Mean Absolute Error of our Model is {}".format(round(score, 3)))


# In[29]:


# Create scatterplot with regression line
sns.regplot(x=Y_pred['Actual'], y=Y_pred['Predicted'], color='blue', scatter_kws={'s': 20, 'alpha': 0.6, 'color': 'red'}, line_kws={'lw': 1})

# Set the x and y limits of the plot
ruler_length_x,ruler_length_y=(Y_pred['Actual'].min()), (Y_pred['Actual'].max())
plt.xlim(ruler_length_x,ruler_length_y)
plt.ylim(ruler_length_x,ruler_length_y)

# Add a 45-degree line in green
plt.plot([ruler_length_x,ruler_length_y], [ruler_length_x,ruler_length_y], color='green', linestyle='--')

# Add axis labels
plt.xlabel('Actual')
plt.ylabel('Predicted')

#apply grids
plt.grid(True)

# Set the title of the plot
plt.title('Actual vs Predicted')

# Display the plot
plt.show()


# In[30]:


sns.distplot(Y_pred['Actual']-Y_pred['Predicted'])
plt.grid(True)


# ## Decison Tree Model
# 
# ![image-3.png](attachment:image-3.png)

# In[31]:


# create a regressor object
decision_tree_regressor = DecisionTreeRegressor(random_state = 0) 
  
# fit the regressor with X and Y data
decision_tree_regressor.fit(X_train, Y_train)


# In[32]:


Y_pred=decision_tree_regressor.predict(X_test) # Use the trained model to make predictions on the test data set
Y_pred # Print the predicted values


# In[33]:


Y_pred=Y_pred.flatten() # Flatten the predicted values array
Y_pred=np.absolute(Y_pred) # Take the absolute values of the predicted values
Y_pred=pd.DataFrame(Y_pred) # Convert the predicted values to a Pandas DataFrame


# In[34]:


Y_pred['Actual']=Y_test.to_numpy() # Add the actual 'meantemp' values from the test data set to the DataFrame
Y_pred.rename(columns={0:'Predicted'},inplace=True) # Rename the column containing the predicted values


# In[35]:


Y_pred


# In[36]:


accuracy=r2_score(Y_pred['Actual'],Y_pred['Predicted'])*100 # Calculate the accuracy of the linear regression model
print(" Accuracy of the model is %.3f" %abs(accuracy)) # Print the accuracy of the model as a percentage


# In[37]:


# Calculate root mean squared error of predicted and actual values
score = np.sqrt(mean_squared_error(Y_pred['Actual'],Y_pred['Predicted']))

# Print the result using formatted string
print("The Mean Squared Error of our Model is {}".format(round(score, 3)))


# In[38]:


# Calculate mean absolute error of predicted and actual values
score = np.sqrt(mean_absolute_error(Y_pred['Actual'],Y_pred['Predicted']))

# Print the result using formatted string
print("The Mean Absolute Error of our Model is {}".format(round(score, 3)))


# In[39]:


# Create scatterplot with regression line
sns.regplot(x=Y_pred['Actual'], y=Y_pred['Predicted'], color='blue', scatter_kws={'s': 20, 'alpha': 0.6, 'color': 'red'}, line_kws={'lw': 1})

# Set the x and y limits of the plot
ruler_length_x,ruler_length_y=(Y_pred['Actual'].min()), (Y_pred['Actual'].max())
plt.xlim(ruler_length_x,ruler_length_y)
plt.ylim(ruler_length_x,ruler_length_y)

# Add a 45-degree line in green
plt.plot([ruler_length_x,ruler_length_y], [ruler_length_x,ruler_length_y], color='green', linestyle='--')

# Add axis labels
plt.xlabel('Actual')
plt.ylabel('Predicted')

#apply grids
plt.grid(True)

# Set the title of the plot
plt.title('Actual vs Predicted')

# Display the plot
plt.show()


# In[40]:


sns.distplot(Y_pred['Actual']-Y_pred['Predicted'])
plt.grid()


# In[41]:


'''plt.figure(figsize=(120,80))
tree.plot_tree(decision_tree_regressor, filled=True, fontsize=10)
plt.show()'''


# This plot gives the internal visualisation of the decision tree model.

# ## Random Forest Model
# 
# ![image.png](attachment:image.png)

# In[42]:


random_forest_regressor = RandomForestRegressor() # Create a linear regression object
random_forest_regressor.fit(X_train, Y_train) # Fit the model to the training data set


# In[43]:


len(random_forest_regressor.estimators_)


# In[44]:


Y_pred=random_forest_regressor.predict(X_test) # Use the trained model to make predictions on the test data set
Y_pred # Print the predicted values


# In[45]:


Y_pred=Y_pred.flatten() # Flatten the predicted values array
Y_pred=np.absolute(Y_pred) # Take the absolute values of the predicted values
Y_pred=pd.DataFrame(Y_pred) # Convert the predicted values to a Pandas DataFrame


# In[46]:


Y_pred['Actual']=Y_test.to_numpy() # Add the actual 'meantemp' values from the test data set to the DataFrame
Y_pred.rename(columns={0:'Predicted'},inplace=True) # Rename the column containing the predicted values


# In[47]:


Y_pred


# In[48]:


accuracy=r2_score(Y_pred['Actual'],Y_pred['Predicted'])*100 # Calculate the accuracy of the linear regression model
print(" Accuracy of the model is %.3f" %abs(accuracy)) # Print the accuracy of the model as a percentage


# In[49]:


# Calculate root mean squared error of predicted and actual values
score = np.sqrt(mean_squared_error(Y_pred['Actual'],Y_pred['Predicted']))

# Print the result using formatted string
print("The Mean Squared Error of our Model is {}".format(round(score, 3)))


# In[50]:


# Create scatterplot with regression line
sns.regplot(x=Y_pred['Actual'], y=Y_pred['Predicted'], color='blue', scatter_kws={'s': 20, 'alpha': 0.6, 'color': 'red'}, line_kws={'lw': 1})

# Set the x and y limits of the plot
ruler_length_x,ruler_length_y=(Y_pred['Actual'].min()), (Y_pred['Actual'].max())
plt.xlim(ruler_length_x,ruler_length_y)
plt.ylim(ruler_length_x,ruler_length_y)

# Add a 45-degree line in green
plt.plot([ruler_length_x,ruler_length_y], [ruler_length_x,ruler_length_y], color='green', linestyle='--')

# Add axis labels
plt.xlabel('Actual')
plt.ylabel('Predicted')

#apply grids
plt.grid(True)

# Set the title of the plot
plt.title('Actual vs Predicted')

# Display the plot
plt.show()


# In[51]:


sns.distplot(Y_pred['Actual']-Y_pred['Predicted'])
plt.grid()


# In[52]:


from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse


# In[53]:


# Additive Decomposition
add_result = seasonal_decompose(Y_train, model = 'additive', period=365, extrapolate_trend = 'freq')
# Multiplicative Decomposition 
mul_result = seasonal_decompose(Y_train, model='multiplicative',period=1)


# In[54]:


add_result.plot().suptitle('Additive Decompose', fontsize=12)
plt.show()


# In[55]:


seasonality=add_result.seasonal
seasonality.plot(color='green')


# In[56]:


mul_result.plot().suptitle('Multiplicative Decompose', fontsize=12)
plt.show()


# In[57]:


new_df_add = pd.concat([add_result.seasonal, add_result.trend, add_result.resid, add_result.observed], axis=1)
new_df_add.columns = ['seasoanilty', 'trend', 'residual', 'actual_values']
new_df_add.head()


# In[58]:


from statsmodels.tsa.stattools import adfuller
adfuller_result = adfuller(Y_train, autolag='AIC')
print(f'ADF Statistic: {adfuller_result[0]}')
print(f'p-value: {adfuller_result[1]}')
for key, value in adfuller_result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')


# In[59]:


from pmdarima.arima import auto_arima


# In[60]:


auto_arima(Y_train, seasonal = True, m=12, start_p=0, start_q=0, max_P =5, max_D=5, max_Q=5).summary()


# In[61]:


import statsmodels.api as sm


# In[62]:


df= Y_train.merge(Y_test)


# In[63]:


df


# In[ ]:


arima_model = sm.tsa.statespace.SARIMAX(Y_train, order=(0,0,1), seasonal_order = (1,1,1,111))
arima_result = arima_model.fit()
arima_result.summary()


# In[ ]:


sarima_pred = arima_result.predict(start= len(Y_train), end= len(Y_train)+110, typ='levels').rename('SARIMA Forecast')
Y_train.plot(figsize=(15,6),legend = True)
sarima_pred.plot()


# In[ ]:


sarima_pred


# In[ ]:


# Calculate root mean squared error of predicted and actual values
score = np.sqrt(mean_squared_error(Y_pred['Actual'],sarima_pred))

# Print the result using formatted string
print("The Mean Squared Error of our Model is {}".format(round(score, 3)))


# In[ ]:





# In[ ]:




