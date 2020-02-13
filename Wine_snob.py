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
