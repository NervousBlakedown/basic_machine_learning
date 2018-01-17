#Import Libraries.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the Dataset.
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

#Encoding Categorical Data.

#Encoding the Independent Variable.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap.
X = X[:, 1:]

#Encoding the Dependent Variable.
#labelencoder_y = LabelEncoder()
#y = labelencoder_y.fit_transform(y)

#Splitting data into training set and test set.
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling.
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#Fitting Multiple Linear Regression to the Training Set.
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test Set results.
y_pred = regressor.predict(X_test)

#Building the Optimal Model using Backward Elimination.
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
