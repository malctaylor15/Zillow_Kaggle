
# coding: utf-8

# # Zillow Zestimate Competition 2017
# ## Linear regression algorithm attempt

# In[1]:


import numpy as np #arrays and matrices
import pandas as pd #data analysis
import matplotlib.pyplot as plt #plotting and visualizations
import pprint as pp #tidying up dictionaries (in our case)

get_ipython().magic('matplotlib inline')
#make the matplotlib graphs show in the notebook
import gc #garbage collection, for memory management


# In[2]:


pd.options.display.float_format = '{:,.4f}'.format


# ### Bring In The Data

# In[3]:


train_df_raw= pd.read_csv("train_2016_v2.csv", parse_dates=["transactiondate"]) #need the file to be in the same directory level as this notebook file
prop = pd.read_csv("properties_2016.csv") #, usecols=good_import_cols) #the properties of each house, where each house is known as parcelid in the data


# ### Remove and Impute Nulls
# ##### Impute = "assign (a value) to something by inference from the value of the products or processes to which it contributes."

# In[4]:



train_df_merged = train_df_raw.merge(prop, on='parcelid', how = 'left')

missing_df = train_df_merged.isnull().sum(axis=0).reset_index() # Finds the number of missing values per column in property file and reset index for the next operation

missing_df.columns = ['column_name', 'missing_count'] # Assign names for the only 2 columns

missing_df = missing_df.loc[missing_df['missing_count']>0] # Remove any features that have no empty rows aka any columns completely filled
missing_df['missing_ratio'] = missing_df['missing_count'] / train_df_merged.shape[0] # Create new column for ratio of empties
missing_df_over75 = missing_df.loc[missing_df['missing_ratio']>0.75] # Remove any with less than 75% vals missing
print(missing_df.sort_values('missing_count', ascending=False)) # Printed to compare the cols that were dropped by the last action
missing_df_over75.sort_values('missing_count', ascending=False)


# In[5]:


train_df_merged.shape


# We see here that some number of cols were dropped by our limit of 75% missing vals. Let's see exactly how many.

# In[6]:


print("Total num of variables: ",train_df_merged.shape[1])
print("Total num of vars w/ all values: ", train_df_merged.drop(missing_df.column_name,axis=1).shape[1])
print("Total num of variables with at least 1 missing value: ", missing_df.shape[0], "\nTotal num of variables w/ at least 1 missing value while also missing at least 75% of its observations: ", missing_df_over75.shape[0]) 
# The shape of the two tuples
print("Total num of variables missing less than 75% of its observations:",missing_df.shape[0] - missing_df_over75.shape[0]) # The num of vars with less than 75% of missing vals


# In[7]:


# Make parcel id the index 
train_df_merged.index = train_df_merged.parcelid


# In[8]:


train_df_dropna = train_df_merged.drop(missing_df_over75.column_name.tolist(), axis = 1) # Remove the cols with > 75% missing vals
train_df_dropna = train_df_dropna.drop(['parcelid'],axis = 1)
train_df_dropna # The remaing NaN vals are where data is missing


# ### Impute the mean where columns are still missing vals 

# In[9]:


mean_values = train_df_dropna.mean(axis=0) #returns Series
train_df_filledna = train_df_dropna.fillna(mean_values, inplace=True) # The actual imputing
train_df_filledna


# Notice how the vals previously NaN are now filled in. The values used are the mean of the other values for each feature aka each column.

# In[10]:


# Add the month to the data frame 

train_df_filledna['transaction_month'] = np.float64(train_df_filledna['transactiondate'].dt.month.values) # Extract and convert the month portion of date to a float value
group_train_months = train_df_filledna.groupby(by=['transaction_month']).logerror.count() # Calculates the number of given results based on the month vals just extracted
group_train_months


# Let's plot this data to get a better idea of it.

# In[11]:


npFloat_group_months = np.float64(group_train_months.index)


# In[12]:


list_group_train = group_train_months.values.tolist()


# In[13]:


fig, ax = plt.subplots()
bar_width = 0.35

cols = npFloat_group_months
counts = list_group_train

barplot = plt.bar(cols, counts)
plt.xlabel('Month')
plt.ylabel('Num Observations')
plt.title('Num Observations by Month')


# Summer months appear to be high times of selling. The last few months are skewed by fewer observations.

# The next steps are a method to find variables that may be extreme or categorical.

# In[14]:


numb_unique_per_col = train_df_filledna.T # Transpose the training dataset
numb_unique_per_col = numb_unique_per_col.apply(lambda x: x.nunique(), axis = 1) # Apply function to each column to get num of unique vals 
numb_unique_per_col.sort_values(0, ascending = True)


# Three columns stand out here: assessmentyear, fips, and regionidcounty.

# In[15]:


numb_unique_per_col[numb_unique_per_col == 1]


# In[16]:


train_df_filledna = train_df_filledna.drop(['assessmentyear'],axis=1) # Only one value
train_df_filledna.shape


# In[17]:


train_df_filledna.dtypes[train_df_filledna.dtypes != "float64"].index


# In[18]:


train_df_filledna['propertycountylandusecode']
#found strings


# In[19]:


train_df_filledna['propertycountylandusecode'].nunique()


# In[20]:


train_df_filledna = train_df_filledna.drop(['propertycountylandusecode'],axis=1)
train_df_filledna.shape


# In[21]:


train_df_filledna['propertyzoningdesc']


# In[22]:


train_df_filledna['propertyzoningdesc'].nunique()


# In[23]:


train_df_filledna = train_df_filledna.drop(['propertyzoningdesc'], axis=1)
train_df_filledna.shape


# In[24]:


numb_unique_per_col[numb_unique_per_col == 1]


# In[25]:


train_df_filledna.columns.tolist()


# In[26]:


numb_unique_per_col.sort_values(0, ascending = True)


# In[27]:


train_df_filledna = train_df_filledna.drop(['censustractandblock','regionidcounty','regionidcity','regionidzip','regionidneighborhood','transactiondate'], axis=1)


# In[28]:


import seaborn as sns
sns.set(style="ticks")

sns.pairplot(train_df_filledna[['bathroomcnt','roomcnt','calculatedbathnbr','bedroomcnt','fullbathcnt']])


# In[29]:


train_df_filledna = train_df_filledna.drop(['bathroomcnt','fullbathcnt'], axis=1) # Too correlated


# In[30]:


sns.pairplot(train_df_filledna[['roomcnt','calculatedbathnbr','bedroomcnt']])


# ### The 3 vars are not too correlated to require removal

# In[31]:


train_df_filledna.columns.tolist()


# In[32]:


print("Variables remaining: ",train_df_filledna.shape[1])


# In[33]:


train_df2 = train_df_filledna


# In[34]:


missing_df2 = train_df2.isnull().sum(axis=0).reset_index()
missing_df2.columns = ['count', 'val']
missing_df2[missing_df2['val'] > 0]
# Should be empty because we removed all NaN vals


# In[35]:


train_df2.head()


# ### Dealing with categorical variables

# In[36]:


from sklearn.preprocessing import OneHotEncoder


# In[37]:


one_hot_colnames_before = ["airconditioningtypeid", 
                    "heatingorsystemtypeid", "propertylandusetypeid"]


# In[38]:


train_df2.groupby(by=['airconditioningtypeid']).logerror.count().reset_index()


# In[39]:


train_df2['airConditionFlag'] = np.ceil(train_df2.airconditioningtypeid).apply(lambda x: 1 if x ==2.0 else 0)


# In[40]:


train_df2.airConditionFlag.mean()


# In[41]:


train_df2.groupby(by=['airConditionFlag']).logerror.count().reset_index()


# In[42]:


train_df2.drop(['airconditioningtypeid'],inplace=True,axis=1)
train_df2.columns.tolist()


# In[43]:


train_df2.groupby(by=['heatingorsystemtypeid']).logerror.count().reset_index()


# In[44]:


train_df2['heatingSystemNew'] = np.ceil(train_df2.heatingorsystemtypeid).apply(lambda x: 1 if x == 4.0 else 0)


# In[45]:


train_df2.heatingSystemNew.mean()


# In[46]:


train_df2.groupby(by=['heatingSystemNew']).logerror.count().reset_index()


# In[47]:


train_df2.drop(['heatingorsystemtypeid'],inplace=True,axis=1)
train_df2.columns.tolist()


# In[48]:


train_df2.head()


# # Change code to not scale the categorical variables 20171016 

# ## Flooring and Capping 

# In[49]:


train_y = train_df2['logerror']


# In[50]:


ceil_y = train_y.quantile(q= 0.95)
floor_y = train_y.quantile(q=0.05)


# In[51]:


print(train_y.quantile(q=0.05))
print(train_y.quantile(q=0.95))


# In[52]:


train_df2['logerror'].describe()


# In[53]:


plt.figure(figsize=(8,6))
plt.scatter(range(train_df2['logerror'].size), np.sort(train_df2['logerror']))
plt.xlabel('index', fontsize=12)
plt.ylabel('logerror', fontsize=12)
plt.axhline(y = ceil_y, linewidth = 2, color = 'r')
plt.axhline(y = floor_y, linewidth = 2, color = 'r')
plt.legend(['values', 'extreme values'])
plt.show()


# In[54]:


train_y[train_y > ceil_y] = ceil_y
train_y[train_y < floor_y] = floor_y


# In[55]:


plt.figure(figsize=(8,6))
plt.hist(train_y, bins = 30)
plt.xlabel('logerror', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()


# In[56]:


plt.figure(figsize=(8,6))
plt.scatter(range(train_y.size), np.sort(train_y))
plt.xlabel('index', fontsize=12)
plt.ylabel('logerror', fontsize=12)
plt.show()


# In[57]:


print("Min value: ",min(train_y))
print("Max value: ", max(train_y)) 


# ### Last minute variable drops 

# In[58]:


train_df2.drop(['transaction_month', 'taxamount', 'finishedsquarefeet12'], axis = 1, inplace = True)


# In[59]:


train_df2.columns


# ### Scale Data

# In[60]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# In[61]:


scaler = StandardScaler().fit(train_df2.drop(['logerror'],axis = 1))


# In[62]:


train_df3 = scaler.transform(train_df2.drop(['logerror'], axis =1))
train_df3.shape


# In[63]:


train_df4 = pd.DataFrame(train_df3,columns=train_df2.drop(['logerror'],axis = 1).columns,index=train_df2.index)
train_df4['logerror'] = train_df2['logerror']
train_df4.head()

#train_df_modeling['logerror'] = train_y
#train_df_modeling.to_csv("semi_clean_df.csv")
# ## Linear Regression

# ### Start by splitting into test train set 

# In[64]:


from sklearn.model_selection import train_test_split


# In[65]:


X = train_df4.drop(['logerror'], axis = 1)
y = train_y


# In[66]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[67]:


print(X.shape)
print(y.shape)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[68]:


import statsmodels.api as sm


# In[69]:


model = sm.OLS(y_train, X_train)
results = model.fit()
print(results.summary())


# In[70]:


results.params[abs(results.params) > 0.02]


# In[71]:


results.pvalues[results.pvalues < 0.05]

results.pvalues[results.pvalues < 0.05] results.params[abs(results.params) > 0.02]
# In[72]:


print("Number of variables without pvalue issues: ", len(results.pvalues[results.pvalues < 0.05]))
print("Number of variables with relatively significant coefficients: ", len(results.params[abs(results.params) > 0.02]))


# ## Examine Validation Set 

# In[73]:


test_pred = results.predict(X_test)


# In[74]:


from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score, roc_curve


# In[75]:


r2_dev = r2_score(y_train, results.fittedvalues)
r2_dev


# In[76]:


r2_itv = r2_score(y_test, test_pred)
r2_itv


# In[77]:


(r2_dev/r2_itv - 1)*100


# In[78]:


mean_absolute_error(y_test, test_pred)


# In[79]:


# Scaled mae 
(mean_absolute_error(y_test, test_pred)/y_test.shape)


# In[80]:


plt.scatter(y_test, test_pred)
plt.xlabel("Actual")
plt.ylabel("Model Predictions")
plt.title("Actuals vs Model Predictions")


# In[81]:


results = pd.DataFrame(data = {"y_test": y_test, "test_pred": test_pred})


# In[82]:


fig, ax = plt.subplots()
ax.scatter(results.y_test[abs(results.test_pred) < 0.25], results.test_pred[abs(results.test_pred) < 0.25])

lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]

# now plot both limits against eachother
ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

plt.xlabel("Actual")
plt.ylabel("Model Predictions")
plt.title("Actual vs Model Predictions")


# In[83]:


#sns.pairplot(train_df4)

### VIF Analysis  
We may want to hold this until later when we have a smaller model..from statsmodels.stats.outliers_influence import variance_inflation_factorvif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]print(len(vif))
print(len(X.columns))vif_series = pd.Series(data = vif, index = X.columns)vif_series.sort_values()vif_series.describe()vif_series[vif_series < 5]inf_vif_list = vif_series[vif_series == np.inf].indexvif_series1 = vif_series.drop(inf_vif_list.tolist())len(vif_series1)vif_series1.head()X_vif1 = X.drop(inf_vif_list.tolist(), axis =1)X_vif1.shapenumb_X_cols = X_vif1.shape[1]
X_ndarray_vif1 =X_vif1.values
vif1 = [variance_inflation_factor(X_ndarray_vif1, i) for i in range(numb_X_cols)]vif1_series = pd.Series(data = vif1, index = X_vif1.columns) vif1_series.describe()
# ### Submission Steps

# We will want to make the dataset as similar to the training set as possible before running the predictive steps  
# * We will have to drop the variables that had too many missing training examples
# * Create the flags for the variables we made flags for 
# * Potentially work with the date time column 
# * Run through the standard scaler 
# * Finally, run through the predictive algorithm 

# In[84]:


train_df_new = pd.read_csv("train_2016_v2.csv")
test = pd.read_csv("sample_submission.csv")
prop = pd.read_csv("properties_2016.csv")


# In[85]:


train_df_new.head()


# In[86]:


test.head()


# In[87]:


prop.head()


# Create the master test set by combining parcelid's specified in the submission file with the parcel id from the properties file  
# 
# We want to change the ParcelId from the test set to match the parcelid column from the properties file.  
# Without changing it, they will not match correctly 

# In[88]:


test['parcelid'] = test['ParcelId']
df_test = test.merge(prop, on='parcelid', how = 'left') 
print("Test DF len: " ,len(df_test))
print("Input DF len: ", len(test))


# In[89]:


df_test.head()


# Start by dropping variables that had too many missing values in the training set 

# In[90]:


df_test1 = df_test.drop(missing_df_over75.column_name.tolist(), axis = 1)


# Impute mean for the missing NAs, we will use the means previously found from the training set  

# In[91]:


df_test_filled_na = df_test1.fillna(mean_values, inplace=True)


# Drop more columns  
# Create flag variable for missing observations 

# In[92]:


df_test2 = df_test_filled_na.drop(['censustractandblock','regionidcounty',
                        'regionidcity','regionidzip','regionidneighborhood',
                         'assessmentyear' , 'propertycountylandusecode',
                        'propertyzoningdesc', 'bathroomcnt','fullbathcnt', 'ParcelId',
                        'airconditioningtypeid', 'finishedsquarefeet12', 'heatingorsystemtypeid',
                        'taxamount'], axis=1)
df_test2.index = df_test2.parcelid
df_test2.drop(['parcelid'], axis =1, inplace= True )
#####
# TODO implement logic from below 
#####
df_test2['heatingSystemNew'] = 1
df_test2['airConditionFlag'] = 0
#df_test2['heatingSystemNew'] = np.ceil(df_test_filled_na.heatingorsystemtypeid).apply(lambda x: 1 if x == 4.0 else 0)
#df_test2['airConditionFlag'] = np.ceil(df_test_filled_na.airconditioningtypeid).apply(lambda x: 1 if x == 2.0 else 0)


# Check if datasets have the same variables 

# In[93]:


df_test2.shape


# In[94]:


train_df4.shape


# In[95]:


df_test2.columns


# In[96]:


train_df4.columns


# In[97]:


cols = []
for column in df_test2.columns:
    if column not in train_df4.columns:
        print(column)
        cols.append(column)


# In[98]:


for column in train_df4.columns:
    if column not in df_test2.columns:
        print(column)


# In[99]:


df_test_pred = df_test2.drop(cols, axis=1)


# In[100]:


df_test_pred.columns


# Scale the test dataset using the same parameters as the training dataset 

# In[101]:


scaled_test_df = scaler.transform(df_test_pred)


# ### Use Linear Regression Model to create predictions 

# In[113]:


predictions = model.fit().predict(scaled_test_df)


# ### Begin preparing submission file  

# In[117]:


sub = pd.read_csv("sample_submission.csv")
# Put the same prediction in all the columns
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = predictions


# In[115]:


# Export to csv 
sub.to_csv('LinearRegression.csv', index=False, float_format='%.4f')


# In[116]:


sub.shape
# Should be 7 -- 6 time periods and 1 parcel ID 

