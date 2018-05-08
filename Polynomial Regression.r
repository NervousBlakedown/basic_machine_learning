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

# Fit linear regression to dataset.
lin_reg = lm(formula = Salary ~ ., data = dataset)

# Fit polynomial regression to dataset.
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
poly_reg = lm(formula = Salary ~ ., data = dataset)

# Visualize linear regression results.
install.packages('ggplot2')
library(ggplot2)
ggplot() + geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') + geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = )), colour = 'blue')

# Visualize polynomial regression results.
