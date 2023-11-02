#!/usr/bin/env python
# coding: utf-8

# In[316]:


# import libraries
import pandas as pd


# In[317]:


# read the data in
weather = pd.read_csv("C:\\Users\\ejiro\\Downloads\\warsaw weather.csv", index_col="DATE")


# In[318]:


# check for null values and convert to percentages
weather.apply(pd.isnull).sum().sort_values(ascending = False) / weather.shape[0]


# In[319]:


# create a new dataframe with the necessary columns
warsaw_weather = weather[["PRCP", "SNWD", "TMAX", "TMIN"]].copy()


# In[320]:


# rename the columns 
warsaw_weather.columns = ['precip', 'snow_depth', 'temp_max', 'temp_min']


# In[322]:


# Exploring the data for null values


# Determine the null values:
null_values = warsaw_weather.isnull().sum().to_frame()
null_values = null_values.rename(columns = {0:'null'})

# Determine the not null values:
not_null = warsaw_weather.notna().sum().to_frame()
not_null = not_null.rename(columns = {0:'not null'})

# Combine the dataframes:

null_count = pd.concat([null_values, not_null], ignore_index=False, axis=1).reset_index()
null_count = null_count.rename(columns = {'index':'category'})

# Display the new dataframe:
null_count


# In[323]:


# Based on the above code outputs, it is clear the data contains many null values spread across various rows and columns
# Some columns are more affected than others. The next step for understanding the missing values is visualization

# Data Manipulation to Create a Dataframe and Chart Outputting Null and Not Null Value Counts
import missingno as mi
import plotly.express as px
# Determine the null values:
null_values = warsaw_weather.isnull().sum().to_frame()
null_values = null_values.rename(columns = {0:'Null'})
# Determine the not null values:
not_null = warsaw_weather.notna().sum().to_frame()
not_null = not_null.rename(columns = {0:'Not Null'})
# Combine the dataframes:
null_count = pd.concat([null_values, not_null], ignore_index=False, axis=1).reset_index()
null_count = null_count.rename(columns = {'index':'Category'})
# Generate Plot
fig = px.bar(null_count, x="Category", y = ['Not Null', 'Null'])
fig.update_xaxes(categoryorder='total descending')
fig.update_layout(
    title={'text':"Null Values Visualization",
           'xanchor':'center',
           'yanchor':'top',
           'x':0.5},
    xaxis_title = "Category",
    yaxis_title = "Count")
fig.update_layout(legend_title_text = 'Status')
fig.show()


# In[324]:


# Drop the snow_depth column as it has too many null values
del warsaw_weather["snow_depth"]


# In[63]:


# Handling missing values in individual columns
# We will be using imputation for handling the missing values.
# Imputation is the act of replacing missing data with statistical estimates of the missing values.
# The imputation method should be decided after considering the distribution of data.
# Mean imputation works better if the distribution is normally-distributed or has a Gaussian distribution, 
# while median imputation is preferable for skewed distribution(be it right or left).

# Next step, would involve visualize the distribution of the data


# In[325]:


# import needed libraries
import matplotlib.pyplot as plt
import numpy as np


# In[326]:


data = warsaw_weather['temp_min']

# Create a histogram to visualize the data distribution
plt.hist(data, bins=30, edgecolor='k', alpha=0.7, color='blue')  # Adjust the number of bins as needed
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Data Distribution of Temp_Min')

# Show the plot
plt.show()


# In[327]:


# Since our data is left-skewed, we will be using the median imputation
warsaw_weather["temp_min"] = warsaw_weather["temp_min"].fillna(warsaw_weather["temp_min"].median())


# In[328]:


# check for null values
warsaw_weather["temp_min"].isnull().sum()


# In[329]:


# now, check the distribution of the temp max column
data = warsaw_weather['temp_max']

# Create a histogram to visualize the data distribution
plt.hist(data, bins=30, edgecolor='k', alpha=0.7, color='blue')  # Adjust the number of bins as needed
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Data Distribution of Temp_Max')

# Show the plot
plt.show()


# In[330]:


# As our data is normally distributed, we would be using the mean imputation 
warsaw_weather["temp_max"] = warsaw_weather["temp_max"].fillna(warsaw_weather["temp_max"].mean())


# In[331]:


# check for null values
warsaw_weather["temp_max"].isnull().sum()


# In[332]:


# now, check the distribution of the precip column
data = warsaw_weather['precip']

# Create a histogram to visualize the data distribution
plt.hist(data, bins=30, edgecolor='k', alpha=0.7, color='blue')  # Adjust the number of bins as needed
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Data Distribution of Precip')

# Show the plot
plt.show()


# In[333]:


# Since our data is right-skewed, we will be using the median imputation
warsaw_weather["precip"] = warsaw_weather["precip"].fillna(warsaw_weather["precip"].median())


# In[334]:


warsaw_weather["precip"].isnull().sum()


# In[335]:


warsaw_weather.apply(pd.isnull).sum()


# In[336]:


# Check to see if the value 9999, which indicates a measurement error, is in any of the columns.
warsaw_weather.apply(lambda x: (x==9999).sum())


# In[337]:


# Check to see if the value 9999, which indicates a measurement error, is in any of the columns.
value = 9999

# Check if "9999" exists in any of the columns
if (warsaw_weather == value).any().any():
   print(f"Value {value} exists in at least one column.")
else:
   print(f"Value {value} does not exist in any column.")


# In[342]:


# verify the datatypes
# Ensure that all of the columns are numeric (float or integer).
warsaw_weather[


# In[343]:


# Outlier detection

# calculate IQR for column Height
#Q1 = df['Height'].quantile(0.25)
#Q3 = df['Height'].quantile(0.75)
#IQR = Q3 - Q1

# identify outliers
#threshold = 1.5
#outliers = df[(df['Height'] < Q1 - threshold * IQR) | (df['Height'] > Q3 + threshold * IQR)]

Q1 = warsaw_weather["precip"].quantile(0.25)
Q3 = warsaw_weather["precip"].quantile(0.75)
IQR = Q3 - Q1

#threshold = 1.5
outliers = warsaw_weather[(warsaw_weather['precip'] < Q1 - threshold * IQR) | (warsaw_weather['precip'] > Q3 + threshold * IQR)]
# Calculate the Z-scores for the 'precip' column
#z = (warsaw_weather['precip'] - warsaw_weather['precip'].mean()) / warsaw_weather['precip'].std()

# replace outliers with median value
#warsaw_weather.loc[z > threshold, 'precip'] = warsaw_weather['precip'].median()


warsaw_weather = warsaw_weather.drop(outliers.index)


# In[345]:


warsaw_weather['precip'].plot()


# In[285]:


# now do same for the 2 other columns

Q1_tmp = warsaw_weather["temp_max"].quantile(0.25)
Q3_tmp = warsaw_weather["temp_max"].quantile(0.75)
IQR_m = Q3_temp - Q1_temp

threshold = 1.5
temp_outliers = warsaw_weather[(warsaw_weather['temp_max'] < Q1_tmp - threshold * IQR) | (warsaw_weather['temp_max'] > Q3_tmp + threshold * IQR_m)]
# Calculate the Z-scores for the 'precip' column
tmp_z = (warsaw_weather['temp_max'] - warsaw_weather['temp_max'].mean()) / warsaw_weather['temp_max'].std()

# replace outliers with median value
warsaw_weather.loc[z > threshold, 'temp_max'] = warsaw_weather['temp_max'].median()


# In[346]:


#temp_outliers.plot()
warsaw_weather[["temp_min", "temp_max"]].plot()


# In[45]:


# Verify the index is stored as a datetime. If it isn't, then convert it.
warsaw_weather.index = pd.to_datetime(warsaw_weather.index)
# now you can check the index by month or year
warsaw_weather.index.year


# In[ ]:





# In[47]:


# next, we will need to start analyzing the data 
# Use pandas to plot the TMIN and TMAX columns. Do you see any interesting trends?

warsaw_weather[["temp_max", "temp_min"]].plot()


# In[48]:


# we can check which years are missing in our dataset
warsaw_weather.index.year.value_counts().sort_index()
# no year is missing


# In[348]:


# plot the precip and examine
warsaw_weather["precip"].plot()


# In[350]:


# you can examine further how much it rained each year by grouping by year
warsaw_weather.groupby(warsaw_weather.index.year).sum()["precip"]


# In[51]:


# first thing is deciding what to predict

# Training an initial model
# We'll be predicting tomorrow's temperature given historical data. 
# In order to do this, we need to create a target column, then create a train and test set and train a model.


warsaw_weather["target"] = warsaw_weather.shift(-1)["temp_max"]
# shift(-1) will pull every row back one position
# so, the code above basically predicts tomorrow's temperature using data for today


# In[52]:


warsaw_weather


# In[131]:


# The last row won't have a next day, so you won't have a target. Remove the last row in the data.
warsaw_weather = warsaw_weather.iloc[:-1,:].copy()


# In[132]:


# Initialize a machine learning model. We recommend using a ridge regression.
# Ridge reg is a type of reg that reduces overfitting.

from sklearn.linear_model import Ridge
reg = Ridge(alpha=.1)

# alpha is howmuch of the coefficients of the reg model are penalized. the greater the penalty, the more
# overfitting is prevented.


# In[133]:


#create predictors we will be using
predictors = ["precip", "temp_max", "temp_min"]


# In[134]:


#Split the data into training and test sets, respecting the order of the data.
train = warsaw_weather.loc[:"2023-08-31"]
test = warsaw_weather.loc["2023-09-01":"2023-12-31"]

# this split is important coz you don;t want to use data from the future to predict the past


# In[135]:


train


# In[136]:


test


# In[137]:


# call the fit model on our reg model and fit it to our training data and fit it to our predictors
reg.fit(train[predictors], train['target'])


# In[ ]:





# In[138]:


# generate predictions on our test data using the predictor column
predictions = reg.predict(test[predictors])


# In[139]:


# check how well we did
from sklearn.metrics import mean_absolute_error
# mean_absolute_error subtracts the actual from the predictions, takes the absolute value and then finds
# the average of that across all of the predictions

mean_absolute_error(test["target"], predictions)


# In[140]:


# Evaluating our model
# first let's combine our actuals and our predictions
# we can view our actual values and our predictions side by side using this code to see the
# where we have big differences and what not
combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
combined.columns = ["actual", "predictions"]


# In[141]:


combined


# In[142]:


# now we can use this to diagnose any issues by plotting it out. we can see cases where we are way off using 
combined.plot()


# In[143]:


# next, we check the coefficients of the regression model to see how the different variables are used by the model
reg.coef_


# In[144]:


# Write a function to finish splitting the data, training a model, and making predictions.
# Ensure the function returns a single DataFrame with both the predictions and the actual values.


# creating a function to make predictions
def creating_predictions(predictors, core_weather, reg):
    train = core_weather.loc[:"2023-08-31"]
    test = core_weather.loc[:"2021-09-01"]
    reg.fit(train[predictors], train['target'])
    predictions = reg.predict(test[predictors])
    error = mean_absolute_error(test["target"], predictions)
    combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
    combined.columns = ["actual", "predictions"]
    return error, combined
    
# what the function does is, instead of reiterating the code everytime, we cam just iterate our model with the function, add more 
# predictors, try testing different models a lot more easily instead of pasting the codes each time


# In[145]:


# adding in rolling means
# let's add more predictors
warsaw_weather["month_max"] = warsaw_weather["temp_max"].rolling(30).mean()
# here we using the rolling function to find a rolling mean for a certain period
# so, the code looks back 30 days from each day and just find the average of a certain column during that period


# In[112]:


warsaw_weather
# some rows are NaN because you actually need 30 days before a row to get the rolling mean


# In[146]:


# now we can find some interesting ratios.
# for a given day we can check if the temperature was different from the monthly mean
# we do this by looking at the monthly mean temperature and divide it by the temperature in that day
warsaw_weather["month_day_max"] = warsaw_weather["month_max"] / warsaw_weather["temp_max"]


# In[147]:


warsaw_weather


# In[148]:


# we can also look at the maximum ratio btw the max and min temperature
warsaw_weather["max_min"] = warsaw_weather["temp_max"] / warsaw_weather["temp_min"]


# In[149]:


# we can use these new columns to update our predictors
predictors = ["precip", "temp_max", "temp_min", "month_max", "month_day_max", "max_min"]


# In[150]:


# these removes the NaN values as it would throw an error if we don't remove them
warsaw_weather = warsaw_weather.iloc[30:,:].copy()


# In[261]:


# drop all infinite values on the dataset 
warsaw_weather = warsaw_weather[~warsaw_weather.isin([np.inf, -np.inf]).any(axis=1)]
#error, combined = creating_predictions(predictors, warsaw_weather, reg)


# In[194]:


error


# In[195]:


combined.plot()


# In[196]:


# adding in monthly and daily averages
# we can try to figure out the historical monthly average in a given month
warsaw_weather["monthly_avg"] = warsaw_weather["temp_max"].groupby(warsaw_weather.index.month).apply(lambda x: x.expanding(1).mean())
# this basically groups all the temperatures for jan together, same for feb etc regardless of what year they fell in, then we 
# applied a function to take the mean of the values before a given day. what we don't want to do is take the mean of all the months
# the expanding function takes only takes the previous rows and calculates the mean
# this way we prevent using future knowledge to influence our past


# In[197]:


warsaw_weather


# In[198]:


# we can also get the day of year average
warsaw_weather["day_of_year_avg"] = warsaw_weather["temp_max"].groupby(warsaw_weather.index.day_of_year).apply(lambda x: x.expanding(1).mean())


# In[199]:


error, combined = creating_predictions(predictors, warsaw_weather, reg)


# In[200]:


error


# In[201]:


# running model diagnostics

# we can take a look at the coefficients of the regression model to see which variables are being used and which aren't
reg.coef_


# In[203]:


# also, we can take a look at the correlations just to see which columns corelate with our target
# this helps us diagnose and see if the predictors were passing in
warsaw_weather.corr()["target"]

# looking at correlations is one good way at taking a look at which predictors you want to use and which one not to


# In[204]:


# another thing we can look at if we want to improve this model is to look at the difference btw the actual value and predicted value
combined["diff"] = (combined["actual"] - combined["predictions"]).abs()
# this will give the absolute difference btw the actual value and comboined


# In[207]:


# next, we look at the biggest differences btw what the actual temperature was and what we predicted
combined.sort_values("diff", ascending=False).head()


# In[208]:


combined


# In[209]:


error


# In[ ]:




