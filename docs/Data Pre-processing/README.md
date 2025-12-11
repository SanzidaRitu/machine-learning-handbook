# Data Pre-processing

This section covers techniques and best practices for preparing data for machine learning models.

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

# Optionally, we can specify exactly how many top rows to print
df.head(10)
```

## Print Some Bottom Rows From The Dataset
```python
# Print some bottom rows (by default prints 5)
df.tail()

# Optionally, we can specify exactly how many bottom rows to print
df.tail(10)
```
## Check the sanity of a dataset
```python
# Check the shape
df.shape

# Check the information
df.info()

# Count the missing values
df.isnull().sum()

# Find the percentage of null values
df.isnull().sum()/len(df)*100

# Count the duplicate values
df.duplicated().sum()

# Count the garbage values
for i in df.select_dtypes(include = "object").columns:
  print(df[i].value_counts())
```

## Exploratory Data Analysis(EDA) to understand data before building models
```python
# Remove leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Check the descriptive statistics
df.describe().T

# Check the discriptive statistics of object
df.describe(include = "object").T

# Check the histogram to understand the distribution
for i in df.select_dtypes(include = "number").columns:
  sns.histplot(data = df, x = i)
  plt.show()

# Boxplot to identify the outliers
for i in df.select_dtypes(include = "number").columns:
  sns.boxplot(data = df, x = i)
  plt.show()

# Scatterplot to understand the relationship
for i in df.select_dtypes(include = "number").columns:
       sns.scatterplot(data = df, x = i, y = "Life expectancy")
       plt.show()

# Heatmap to understand correlation
plt.figure(figsize=(10,10))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True)
plt.show()
```

## Missing value treatment
``` python
# Checking missing value columns
df.isnull().sum()

# Fill missing valuesfor numerical columns with median
for i in ['BMI', 'Polio', 'Income composition of resources']: # Column names
  df[i] = df[i].fillna(df[i].median())

df.isnull().sum() # Checking missing value columns

# Fill missing values using KNNImputer
from sklearn.impute import KNNImputer # Import necessary libraries
imputer = KNNImputer()

for i in df.select_dtypes(include='number').columns: #Apply the imputer to fill missing values in each numeric column
  df[i] = imputer.fit_transform(df[[i]])

df.isnull().sum() # Checking missing value columns
```

## Outliers treatment
```python
# Define whisker function to detect, flag, or handle outliers consistently
def wisker(col):
  q1, q3 = np.percentile(col, [25, 75])
  iqr = q3 - q1
  lw = q1 - (1.5 * iqr)
  uw = q3 + (1.5 * iqr)
  return lw, uw

# Replace the outliers with lower wisker and upper wisker
for i in ['GDP', 'Total expenditure', 'thinness  1-19 years', 'thinness 5-9 years']: # Column name
  lw, uw = wisker(df[i])
  df[i] = np.where(df[i] < lw, lw, df[i])
  df[i] = np.where(df[i] > uw, uw, df[i])

# Create a boxplot for the current column
for i in ['GDP', 'Total expenditure', 'thinness  1-19 years', 'thinness 5-9 years']: # Column name
  sns.boxplot(data=df, x=i)
  plt.show()
```

## Duplicates and garbage value treatment
``` python
# Dropping duplicate values
df.drop_duplicates()
```

## Encoding the data
``` python
#Label encoding or one hot encoding using pd.getdummies to convert categorical (non-numeric) data into numeric values
dummy = pd.get_dummies(data = df, columns = ['Country', 'Status'], drop_first = True) # Column name
dummy
```