# Polynomial Regression.

# Import libraries.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset.
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Split dataset into training set and test set.
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)"""

""" Feature scaling.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)"""

# Fit linear regression model to dataset.
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fit polynomial regression model to dataset.
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualizing linear regression results.
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression Results)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing polynomial regression results.
    # X_grid = np.arange(min(X), max(X), 0.1)
    # X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
    # plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression Results)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predict new result with linear regression.
lin_reg.predict(6.5)

# Predict new result with polynomial regression.
lin_reg_2.predict(poly_reg.fit_transform(6.5))
