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
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ ., data = dataset)

# Visualize linear regression results.
install.packages('ggplot2')
library(ggplot2)
ggplot() + geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') +
geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = )), colour = 'blue') +
ggtitle('Truth or Bluff (Linear Regression)') +
xlab('Level') +
ylab('Salary')

# Visualize polynomial regression results.
install.packages('ggplot2')
library(ggplot2)
ggplot() + geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') +
geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = )), colour = 'blue') +
ggtitle('Truth or Bluff (Polynomial Regression)') +
xlab('Level') +
ylab('Salary')

# Predict new result with linear regression.
y_pred = predict(lin_reg, data.frame(Level = 6.5))

# Predict new result with polynomial regression.
y_pred = predict(poly_reg, data.frame(Level = 6.5, Level2 = 6.5^2, Level3 = 6.5^3, Level4 = 6.5^4))
