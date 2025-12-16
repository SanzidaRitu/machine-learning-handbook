# Model Training

This section covers the process of training machine learning models, including algorithms, evaluation, and optimization.

## Contents

## Importing Necessary Libraries For Data Pre-processing
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

## Reading Dataset
```python
# Read a CSV file
df = pd.read_csv('data.csv')
```

## Reading Dataset From Google Drive To Google Collab
```python
# Import library for connecting to google drive
from google.colab import drive

# Connect to drive and grant access permission
drive.mount('/content/drive', force_remount=True)

# Read the dataset from drive
df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Datasets/Life Expectancy Data.csv');
```
## Print Some Top Rows From The Dataset
```python
# Print some top rows (by default prints 5)
df.head()
```

## Data Preparation
```python
# Data separation as X and y
# # Display the target variable
y = df['logS']
y

# Create the feature set (X) by removing the target column
X = df.drop('logS', axis=1)
X
```

## Data Splitting
```python
# Import the train-test split function
from sklearn.model_selection import train_test_split
# Split X and y into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Training feature set
X_train

# Testing feature set
X_test
```

## Model Building
```python
# Linear Regression
# Import the Linear Regression model
from sklearn.linear_model import LinearRegression

# Create a Linear Regression model object
lr = LinearRegression()
# Train the model using the training data
lr.fit(X_train, y_train)

# Predict target values for the training data
y_lr_train_pred = lr.predict(X_train)
# Predict target values for the testing data
y_lr_test_pred = lr.predict(X_test)

# Predicted target values for the training data
y_lr_train_pred

# Predicted target values for the testing data
y_lr_test_pred
```

## Evaluate model performance
```python
# Import model evaluation metrics
from sklearn.metrics import mean_squared_error, r2_score

# Calculate training Mean Squared Error (MSE)
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
# Calculate training R² score
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

# Calculate testing Mean Squared Error (MSE)
lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
# Calculate testing R² score
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

# Print training and testing performance metrics
print('LR MSE (Train): ', lr_train_mse)
print('LR R2 (Train): ', lr_train_r2)
print('LR MSE (Test): ', lr_test_mse)
print('LR R2 (Test): ', lr_test_r2)
```

## Random Forest
```python
# Import the Random Forest regression model
from sklearn.ensemble import RandomForestRegressor

# Create the Random Forest model
rf = RandomForestRegressor(max_depth=2, random_state=100)
# Train the model
rf.fit(X_train, y_train)

# Predict target values for the training data
y_rf_train_pred = rf.predict(X_train)
# Predict target values for the testing data
y_rf_test_pred = rf.predict(X_test)
```

## Evaluate model performance
```python
# Import regression evaluation metrics
from sklearn.metrics import mean_squared_error, r2_score

# Calculate training Mean Squared Error
rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
# Calculate training R² score
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

# Calculate testing Mean Squared Error
rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
# Calculate testing R² score
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

# Create a DataFrame to store Random Forest results
rf_results = pd.DataFrame(['Random forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
# Assign column names to the DataFrame
rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
# Display the results
rf_results
```

## Model Comparison
```python
# Create a DataFrame for Linear Regression results
lr_results = pd.DataFrame(['Linear Regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']

# Now combine both models
df_models = pd.concat([lr_results, rf_results], axis=0)
df_models

# Clean up the DataFrame index for a neat, continuous numbering of rows
df_models.reset_index(drop=True)
```

## Data visualization of prediction results
```python
# Import libraries
import matplotlib.pyplot as plt
import numpy as np

# Create a figure and set its size
plt.figure(figsize=(5,5))
# Plot a scatter plot of actual vs predicted values
plt.scatter(x=y_train, y=y_lr_train_pred, c="#7CAE00" ,alpha=0.3)

# Fit a linear trend line
z = np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)

# Plot the regression line
plt.plot(y_train, p(y_train), '#F8766D')

# Label the axes
plt.ylabel('Predict LogS')
plt.xlabel('Experimental LogS')
```