#Import Libraries.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the Dataset.
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

#Taking care of missing data. Using the mean for the missing data.
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3]) #upper-bound not included, must go past; lower-bound included, though
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Encoding categorical data.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Splitting data into training set and test set.
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
