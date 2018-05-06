# Polynomial Regression.
# Import dataset.
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Splitting dataset into training set and test set.
# install.packages('catools')
# library(catools)
# set.seed(123)
# split = sample.split(dataset$Purchased, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling.
# training_set = scale(training_set)
# test_set = scale(test_set)
