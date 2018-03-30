# Multiple Linear Regression.
# Importing the dataset.
dataset = read_csv('50_Startups.csv')
# dataset = dataset, [, 2:3]


# Encoding categorical data.
dataset$State = factor(dataset$State,
                        levels = c('New York', 'California', 'Florida'),
                        labels = c(1, 2, 3)) # corresponds with levels.


# Splitting data into training set and test set.
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# Feature scaling.
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])


# Fitting multiple linear regression to training set.
regressor = lm(formula = Profit ~ .,
                data = training_set)
