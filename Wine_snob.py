import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

# Importing cross validation pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# Import evaluation metrics
from sklearn.metrics import mean_squared_error, r2_score

#Load wine data
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=";")        # ; separator of the data

print (data.head())

print (data.describe())

# Splitting data from target from training features
y = data.quality                    # objective
X = data.drop('quiality', axis = 1) # taking out the goal

# NOTE: Splitting data in Training and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,      # splitting data in 20% for testting
                                                    random_state=123,
                                                    stratify=y)
''' TRANSFORMER API
    1. Fit the transformer on the training set (saving the means and standard deviations)
    2. Apply the transformer to the training set (scaling the training data)
    3. Apply the transformer to the test set (using the same means and standard deviations)
'''

# Fitting the transfomer API
scaler = preprocessing.StandardScaler().fit(X_train) # Factor para scalar la data entre 0 y 1
