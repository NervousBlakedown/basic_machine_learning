# Naive Bayes.
# Import packages.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import dataset.
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = datset.iloc[:, 4].values

# Split dataset into training set and test set.
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature scaling.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fit classifier to training set.
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

# Predicting test results.
y_pred = classifier.predict(X_test)

# Making confusion matrix.
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualizing training set results.


plt.title('Naive Bayes (Training Set)')








plt.title('Naive Bayes (Test Set)')
