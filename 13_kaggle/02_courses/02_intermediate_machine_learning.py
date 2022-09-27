# Missing values

# Example
# In the example, we will work with the Melbourne Housing dataset. Our model will use information such as the number of rooms and land size to predict home price.
# We won't focus on the data loading step. Instead, you can imagine you are at a point where you already have the training and validation data in X_train, X_valid, y_train, and y_valid.

import pandas as pd
from sklearn.model_selection import train_test_split
import os
path = r'E:\Atriam-Dev\DataScience\DataSciencePython\13_kaggle\02_courses\data'
melbourne_file_path = 'melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
data = pd.read_csv(os.path.join(path,melbourne_file_path))


# Select target
y = data.Price

# To keep things simple, we'll use only numerical predictors
melb_predictors = data.drop(['Price'], axis=1)
melb_predictors.columns
melb_predictors.head()
X = melb_predictors.select_dtypes(exclude=['object'])
X.columns
X.head()
# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

# Define Function to Measure Quality of Each Approach
# We define a function score_dataset() to compare different approaches to dealing with missing values. This function reports the mean absolute error (MAE) from a random forest model.
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


# Score from Approach 1 (Drop Columns with Missing Values)
# Since we are working with both training and validation sets, we are careful to drop the same columns in both DataFrames.
# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

# Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))




# Score from Approach 2 (Imputation)
# Next, we use SimpleImputer to replace missing values with the mean value along each column
# Although it's simple, filling in the mean value generally performs quite well (but this varies by dataset). While statisticians have experimented with more complex ways to determine imputed values (such as regression imputation, for instance), the complex strategies typically give no additional benefit once you plug the results into sophisticated machine learning models.
from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer()
print(my_imputer.fit_transform(X_train))
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

imputed_X_train.head()
imputed_X_train.columns

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE from Approach 2 (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))

# We see that Approach 2 has lower MAE than Approach 1, so Approach 2 performed better on this dataset.



# Score from Approach 3 (An Extension to Imputation)
# Next, we impute the missing values, while also keeping track of which values were imputed.
# Make copy to avoid changing original data (when imputing)
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

print("MAE from Approach 3 (An Extension to Imputation):")
print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))

# As we can see, Approach 3 performed slightly worse than Approach 2.
# So, why did imputation perform better than dropping the columns?
# The training data has 10864 rows and 12 columns, where three columns contain missing data. For each column, less than half of the entries are missing. Thus, dropping the columns removes a lot of useful information, and so it makes sense that imputation would perform better.

# Shape of training data (num_rows, num_columns)
print(X_train.shape)

# Number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

# Conclusion¶
# As is common, imputing missing values (in Approach 2 and Approach 3) yielded better results, relative to when we simply dropped columns with missing values (in Approach 1).






## Categorical Variables

import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data

# Separate target from predictors
y = data.Price
X = data.drop(['Price'], axis=1)

# Divide data into training and validation subsets
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

# Drop columns with missing values (simplest approach)
cols_with_missing = [col for col in X_train_full.columns if X_train_full[col].isnull().any()] 
X_train_full.drop(cols_with_missing, axis=1, inplace=True)
X_valid_full.drop(cols_with_missing, axis=1, inplace=True)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = low_cardinality_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

X_train.head()

# Get list of categorical variables
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)

# Categorical columns in the training data
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

# Columns that can be safely ordinal encoded
good_label_cols = [col for col in object_cols if 
                   set(X_valid[col]).issubset(set(X_train[col]))]
        
# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(object_cols)-set(good_label_cols))

print('Categorical columns that will be ordinal encoded:', good_label_cols)
print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)


# Get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))

# Print number of unique entries by column, in ascending order
sorted(d.items(), key=lambda x: x[1])



# 1) Drop Categorical Variables
# The easiest approach to dealing with categorical variables is to simply remove them from the dataset. This approach will only work well if the columns did not contain useful information.
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

print("MAE from Approach 1 (Drop categorical variables):")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))


# 2) Ordinal Encoding
# Ordinal encoding assigns each unique value to a different integer.
from sklearn.preprocessing import OrdinalEncoder

# Make copy to avoid changing original data 
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

# Apply ordinal encoder to each column with categorical data
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])

print("MAE from Approach 2 (Ordinal Encoding):") 
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))


# 3) One-Hot Encoding
# One-hot encoding creates new columns indicating the presence (or absence) of each possible value in the original data. To understand this, we'll work through an example.
from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

print("MAE from Approach 3 (One-Hot Encoding):") 
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))


# Which approach is best?¶
# In this case, dropping the categorical columns (Approach 1) performed worst, since it had the highest MAE score. As for the other two approaches, since the returned MAE scores are so close in value, there doesn't appear to be any meaningful benefit to one over the other.

# In general, one-hot encoding (Approach 3) will typically perform best, and dropping the categorical columns (Approach 1) typically performs worst, but it varies on a case-by-case basis.




# Pipelines
# A critical skill for deploying (and even testing) complex models with pre-processing.

import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
# data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

# Separate target from predictors
y = data.Price
X = data.drop(['Price'], axis=1)

# Divide data into training and validation subsets
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

X_train.head()

# Step 1: Define Preprocessing Steps¶
# Similar to how a pipeline bundles together preprocessing and modeling steps, we use the ColumnTransformer class to bundle together different preprocessing steps. The code below:

# imputes missing values in numerical data, and
# imputes missing values and applies a one-hot encoding to categorical data.

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=0)

from sklearn.metrics import mean_absolute_error

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)




## Cross-Validation
# A better way to test your models.

import pandas as pd

# Read the data
#data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                              ('model', RandomForestRegressor(n_estimators=50,
                                                              random_state=0))
                             ])

from sklearn.model_selection import cross_val_score

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)

print("Average MAE score (across experiments):")
print(scores.mean())



## XGBoost
# The most accurate modeling technique for structured data.
# Gradient Boosting¶
# Gradient boosting is a method that goes through cycles to iteratively add models into an ensemble.
# Gradient boosting is a method that goes through cycles to iteratively add models into an ensemble.

# It begins by initializing the ensemble with a single model, whose predictions can be pretty naive. (Even if its predictions are wildly inaccurate, subsequent additions to the ensemble will address those errors.)

# Then, we start the cycle:

# First, we use the current ensemble to generate predictions for each observation in the dataset. To make a prediction, we add the predictions from all models in the ensemble.
# These predictions are used to calculate a loss function (like mean squared error, for instance).
# Then, we use the loss function to fit a new model that will be added to the ensemble. Specifically, we determine model parameters so that adding this new model to the ensemble will reduce the loss. (Side note: The "gradient" in "gradient boosting" refers to the fact that we'll use gradient descent on the loss function to determine the parameters in this new model.)
# Finally, we add the new model to ensemble, and ...
# ... repeat!

import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
#data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price

# Separate data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y)

# In this example, you'll work with the XGBoost library. XGBoost stands for extreme gradient boosting, which is an implementation of gradient boosting with several additional features focused on performance and speed. (Scikit-learn has another version of gradient boosting, but XGBoost has some technical advantages.)

# In the next code cell, we import the scikit-learn API for XGBoost (xgboost.XGBRegressor). This allows us to build and fit a model just as we would in scikit-learn. As you'll see in the output, the XGBRegressor class has many tunable parameters -- you'll learn about those soon!


from xgboost import XGBRegressor

my_model = XGBRegressor()
my_model.fit(X_train, y_train)

my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train)


# We also make predictions and evaluate the model.
from sklearn.metrics import mean_absolute_error

predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))

# Parameter Tuning
# XGBoost has a few parameters that can dramatically affect accuracy and training speed. The first parameters you should understand are:

# n_estimators
# n_estimators specifies how many times to go through the modeling cycle described above. It is equal to the number of models that we include in the ensemble.

# Too low a value causes underfitting, which leads to inaccurate predictions on both training data and test data.
# Too high a value causes overfitting, which causes accurate predictions on training data, but inaccurate predictions on test data (which is what we care about).
# Typical values range from 100-1000, though this depends a lot on the learning_rate parameter discussed below.

# Here is the code to set the number of models in the ensemble:

my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train)

# early_stopping_rounds
# early_stopping_rounds offers a way to automatically find the ideal value for n_estimators. Early stopping causes the model to stop iterating when the validation score stops improving, even if we aren't at the hard stop for n_estimators. It's smart to set a high value for n_estimators and then use early_stopping_rounds to find the optimal time to stop iterating.

# Since random chance sometimes causes a single round where validation scores don't improve, you need to specify a number for how many rounds of straight deterioration to allow before stopping. Setting early_stopping_rounds=5 is a reasonable choice. In this case, we stop after 5 straight rounds of deteriorating validation scores.

# When using early_stopping_rounds, you also need to set aside some data for calculating the validation scores - this is done by setting the eval_set parameter.

# We can modify the example above to include early stopping:
my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)],
             verbose=False)

# If you later want to fit a model with all of your data, set n_estimators to whatever value you found to be optimal when run with early stopping.

# learning_rate
# Instead of getting predictions by simply adding up the predictions from each component model, we can multiply the predictions from each model by a small number (known as the learning rate) before adding them in.

# This means each tree we add to the ensemble helps us less. So, we can set a higher value for n_estimators without overfitting. If we use early stopping, the appropriate number of trees will be determined automatically.

# In general, a small learning rate and large number of estimators will yield more accurate XGBoost models, though it will also take the model longer to train since it does more iterations through the cycle. As default, XGBoost sets learning_rate=0.1.

# Modifying the example above to change the learning rate yields the following code:

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)

# n_jobs
# On larger datasets where runtime is a consideration, you can use parallelism to build your models faster. It's common to set the parameter n_jobs equal to the number of cores on your machine. On smaller datasets, this won't help.

# The resulting model won't be any better, so micro-optimizing for fitting time is typically nothing but a distraction. But, it's useful in large datasets where you would otherwise spend a long time waiting during the fit command.

# Here's the modified example:
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)

# Conclusion
# XGBoost is a leading software library for working with standard tabular data (the type of data you store in Pandas DataFrames, as opposed to more exotic types of data like images and videos). With careful parameter tuning, you can train highly accurate models.




# XGBoost
# Set up code checking
import os
if not os.path.exists("../input/train.csv"):
    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  
    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 
from learntools.core import binder
binder.bind(globals())
from learntools.ml_intermediate.ex6 import *
print("Setup Complete")

import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice              
X.drop(['SalePrice'], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]

# Select numeric columns
numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# One-hot encode the data (to shorten the code, we use pandas)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)


# Step 1: Build model
# Part A
# In this step, you'll build and train your first model with gradient boosting.
# Begin by setting my_model_1 to an XGBoost model. Use the XGBRegressor class, and set the random seed to 0 (random_state=0). Leave all other parameters as default.
# Then, fit the model to the training data in X_train and y_train.
from xgboost import XGBRegressor
# Define the model
my_model_1 = XGBRegressor(random_state=0) # Your code here
# Fit the model
my_model_1.fit(X_train, y_train)
# Check your answer
# Calculate MAE
from sklearn.metrics import mean_absolute_error
# Get predictions
predictions_1 = my_model_1.predict(X_valid) # Your code here
mae_1 = mean_absolute_error(predictions_1, y_valid) # Your code here

# Uncomment to print MAE
print("Mean Absolute Error:" , mae_1)

# Check your answer


# Step 2: Improve the model
# Now that you've trained a default model as baseline, it's time to tinker with the parameters, to see if you can get better performance!
# Begin by setting my_model_2 to an XGBoost model, using the XGBRegressor class. Use what you learned in the previous tutorial to figure out how to change the default parameters (like n_estimators and learning_rate) to get better results.
# Then, fit the model to the training data in X_train and y_train.
# Set predictions_2 to the model's predictions for the validation data. Recall that the validation features are stored in X_valid.
# Finally, use the mean_absolute_error() function to calculate the mean absolute error (MAE) corresponding to the predictions on the validation set. Recall that the labels for the validation data are stored in y_valid.
# In order for this step to be marked correct, your model in my_model_2 must attain lower MAE than the model in my_model_1.
# Define the model
my_model_2 = XGBRegressor(random_state=0, n_estimators=1000, learning_rate=0.05, n_jobs=4) # Your code here
# Fit the model
my_model_2.fit(X_train, y_train) # Your code here
# Get predictions
predictions_2 = my_model_2.predict(X_valid) # Your code here
# Calculate MAE
mae_2 = mean_absolute_error(predictions_2, y_valid) # Your code here
# Uncomment to print MAE
print("Mean Absolute Error:" , mae_2)
# Check your answer



# Step 3: Break the model
# In this step, you will create a model that performs worse than the original model in Step 1. This will help you to develop your intuition for how to set parameters. You might even find that you accidentally get better performance, which is ultimately a nice problem to have and a valuable learning experience!
# Begin by setting my_model_3 to an XGBoost model, using the XGBRegressor class. Use what you learned in the previous tutorial to figure out how to change the default parameters (like n_estimators and learning_rate) to design a model to get high MAE.
# Then, fit the model to the training data in X_train and y_train.
# Set predictions_3 to the model's predictions for the validation data. Recall that the validation features are stored in X_valid.
# Finally, use the mean_absolute_error() function to calculate the mean absolute error (MAE) corresponding to the predictions on the validation set. Recall that the labels for the validation data are stored in y_valid.
# In order for this step to be marked correct, your model in my_model_3 must attain higher MAE than the model in my_model_1
# Define the model
my_model_3 = XGBRegressor(random_state=0, n_estimators=10, learning_rate=0.05, n_jobs=4) 
# Fit the model
my_model_3.fit(X_train, y_train) # Your code here
# Get predictions
predictions_3 = my_model_3.predict(X_valid)
# Calculate MAE
mae_3 = mean_absolute_error(predictions_3, y_valid)
# Uncomment to print MAE
print("Mean Absolute Error:" , mae_3)
# Check your answer



# Data Leakage
# Find and fix this problem that ruins your model in subtle ways.
# Data leakage (or leakage) happens when your training data contains information about the target, but similar data will not be available when the model is used for prediction. This leads to high performance on the training set (and possibly even the validation data), but the model will perform poorly in production.
# In other words, leakage causes a model to look accurate until you start making decisions with the model, and then the model becomes very inaccurate.
# There are two main types of leakage: target leakage and train-test contamination.

# Target leakage
# Target leakage occurs when your predictors include data that will not be available at the time you make predictions. 
# It is important to think about target leakage in terms of the timing or chronological order that data becomes available, not merely whether a feature helps make good predictions.
# An example will be helpful. Imagine you want to predict who will get sick with pneumonia. The top few rows of your raw data look like this:

# got_pneumonia	age	weight	male	took_antibiotic_medicine	...
# False	65	100	False	False	...
# False	72	130	True	False	...
# True	58	100	False	True	...

# People take antibiotic medicines after getting pneumonia in order to recover. 
# The raw data shows a strong relationship between those columns, but took_antibiotic_medicine is frequently changed after the value for got_pneumonia is determined. This is target leakage.
# The model would see that anyone who has a value of False for took_antibiotic_medicine didn't have pneumonia. 
# Since validation data comes from the same source as training data, the pattern will repeat itself in validation, and the model will have great validation (or cross-validation) scores.
# But the model will be very inaccurate when subsequently deployed in the real world, 
# because even patients who will get pneumonia won't have received antibiotics yet when we need to make predictions about their future health.
# To prevent this type of data leakage, any variable updated (or created) after the target value is realized should be excluded.

# Train-Test Contamination
# A different type of leak occurs when you aren't careful to distinguish training data from validation data.
# Recall that validation is meant to be a measure of how the model does on data that it hasn't considered before. 
# You can corrupt this process in subtle ways if the validation data affects the preprocessing behavior. 
# This is sometimes called train-test contamination.
# For example, imagine you run preprocessing (like fitting an imputer for missing values) before calling train_test_split(). 
# The end result? Your model may get good validation scores, giving you great confidence in it, but perform poorly when you deploy it to make decisions.
# After all, you incorporated data from the validation or test data into how you make predictions, 
# so the may do well on that particular data even if it can't generalize to new data. This problem becomes even more subtle (and more dangerous) when you do more complex feature engineering.
# If your validation is based on a simple train-test split, exclude the validation data from any type of fitting, 
# including the fitting of preprocessing steps. This is easier if you use scikit-learn pipelines. When using cross-validation, 
# it's even more critical that you do your preprocessing inside the pipeline!


# Example
# In this example, you will learn one way to detect and remove target leakage.
# We will use a dataset about credit card applications and skip the basic data set-up code. 
# The end result is that information about each credit card application is stored in a DataFrame X. 
# We'll use it to predict which applications were accepted in a Series y.

import pandas as pd

# Read the data
# data = pd.read_csv('../input/AER_credit_card_data.csv', 
#                    true_values = ['yes'], false_values = ['no'])
data = pd.read_csv(os.path.join(path,'AER_credit_card_data.csv'), true_values = ['yes'], false_values = ['no'])

# Select target
y = data.card

# Select predictors
X = data.drop(['card'], axis=1)

print("Number of rows in the dataset:", X.shape[0])
X.head()

# Since this is a small dataset, we will use cross-validation to ensure accurate measures of model quality.
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Since there is no preprocessing, we don't need a pipeline (used anyway as best practice!)
my_pipeline = make_pipeline(RandomForestClassifier(n_estimators=100))
cv_scores = cross_val_score(my_pipeline, X, y, 
                            cv=5,
                            scoring='accuracy')

print("Cross-validation accuracy: %f" % cv_scores.mean())
# With experience, you'll find that it's very rare to find models that are accurate 98% of the time. It happens, but it's uncommon enough that we should inspect the data more closely for target leakage.

# Here is a summary of the data, which you can also find under the data tab:

# card: 1 if credit card application accepted, 0 if not
# reports: Number of major derogatory reports
# age: Age n years plus twelfths of a year
# income: Yearly income (divided by 10,000)
# share: Ratio of monthly credit card expenditure to yearly income
# expenditure: Average monthly credit card expenditure
# owner: 1 if owns home, 0 if rents
# selfempl: 1 if self-employed, 0 if not
# dependents: 1 + number of dependents
# months: Months living at current address
# majorcards: Number of major credit cards held
# active: Number of active credit accounts
# A few variables look suspicious. For example, does expenditure mean expenditure on this card or on cards used before applying?

# At this point, basic data comparisons can be very helpful:
expenditures_cardholders = X.expenditure[y]
expenditures_noncardholders = X.expenditure[~y]

print('Fraction of those who did not receive a card and had no expenditures: %.2f' \
      %((expenditures_noncardholders == 0).mean()))
print('Fraction of those who received a card and had no expenditures: %.2f' \
      %(( expenditures_cardholders == 0).mean()))

# Fraction of those who did not receive a card and had no expenditures: 1.00
# Fraction of those who received a card and had no expenditures: 0.02

# As shown above, everyone who did not receive a card had no expenditures, while only 2% of those who received a card had no expenditures. It's not surprising that our model appeared to have a high accuracy. But this also seems to be a case of target leakage, where expenditures probably means expenditures on the card they applied for.
# Since share is partially determined by expenditure, it should be excluded too. The variables active and majorcards are a little less clear, but from the description, they sound concerning. In most situations, it's better to be safe than sorry if you can't track down the people who created the data to find out more.
# We would run a model without target leakage as follows:
# Drop leaky predictors from dataset
potential_leaks = ['expenditure', 'share', 'active', 'majorcards']
X2 = X.drop(potential_leaks, axis=1)

# Evaluate the model with leaky predictors removed
cv_scores = cross_val_score(my_pipeline, X2, y, 
                            cv=5,
                            scoring='accuracy')

print("Cross-val accuracy: %f" % cv_scores.mean())

# This accuracy is quite a bit lower, which might be disappointing. 
# However, we can expect it to be right about 80% of the time when used on new applications, 
# whereas the leaky model would likely do much worse than that (in spite of its higher apparent score in cross-validation).

# Conclusion
# Data leakage can be multi-million dollar mistake in many data science applications. 
# Careful separation of training and validation data can prevent train-test contamination, 
# and pipelines can help implement this separation. Likewise, a combination of caution, common sense, 
# and data exploration can help identify target leakage.



import pandas as pd

# Read the data

data = pd.read_csv(os.path.join(path,'AER_credit_card_data.csv'), true_values = ['yes'], false_values = ['no'])

# Select target
y = data.card

# Select predictors
X = data.drop(['card'], axis=1)

print("Number of rows in the dataset:", X.shape[0])
X.head()

# Since this is a small dataset, we will use cross-validation to ensure accurate measures of model quality.
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Since there is no preprocessing, we don't need a pipeline (used anyway as best practice!)
my_pipeline = make_pipeline(RandomForestClassifier(n_estimators=100))
cv_scores = cross_val_score(my_pipeline, X, y, 
                            cv=5,
                            scoring='accuracy')

print("Cross-validation accuracy: %f" % cv_scores.mean())


# With experience, you'll find that it's very rare to find models that are accurate 98% of the time. It happens, but it's uncommon enough that we should inspect the data more closely for target leakage.

# Here is a summary of the data, which you can also find under the data tab:

# card: 1 if credit card application accepted, 0 if not
# reports: Number of major derogatory reports
# age: Age n years plus twelfths of a year
# income: Yearly income (divided by 10,000)
# share: Ratio of monthly credit card expenditure to yearly income
# expenditure: Average monthly credit card expenditure
# owner: 1 if owns home, 0 if rents
# selfempl: 1 if self-employed, 0 if not
# dependents: 1 + number of dependents
# months: Months living at current address
# majorcards: Number of major credit cards held
# active: Number of active credit accounts
# A few variables look suspicious. For example, does expenditure mean expenditure on this card or on cards used before applying?

# At this point, basic data comparisons can be very helpful:
expenditures_cardholders = X.expenditure[y]
expenditures_noncardholders = X.expenditure[~y]

print('Fraction of those who did not receive a card and had no expenditures: %.2f' \
      %((expenditures_noncardholders == 0).mean()))
print('Fraction of those who received a card and had no expenditures: %.2f' \
      %(( expenditures_cardholders == 0).mean()))

    
# As shown above, everyone who did not receive a card had no expenditures, while only 2% of those who received a card had no expenditures. It's not surprising that our model appeared to have a high accuracy. But this also seems to be a case of target leakage, where expenditures probably means expenditures on the card they applied for.

# Since share is partially determined by expenditure, it should be excluded too. The variables active and majorcards are a little less clear, but from the description, they sound concerning. In most situations, it's better to be safe than sorry if you can't track down the people who created the data to find out more.

# We would run a model without target leakage as follows:
    
# Drop leaky predictors from dataset
potential_leaks = ['expenditure', 'share', 'active', 'majorcards']
X2 = X.drop(potential_leaks, axis=1)

# Evaluate the model with leaky predictors removed
cv_scores = cross_val_score(my_pipeline, X2, y, 
                            cv=5,
                            scoring='accuracy')

print("Cross-val accuracy: %f" % cv_scores.mean())
    
