# SVR.
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

# Fit SVR to dataset.
install.packages('e1071')
library(e1071)
regressor = svm(formula = Salary ~., data = dataset, type = 'eps-regression')

# Predict new result with linear regression.
y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualize SVR results.
install.packages('ggplot2')
library(ggplot2)
ggplot() + geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') +
geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = )), colour = 'blue') +
ggtitle('Truth or Bluff (SVR)') +
xlab('Level') +
ylab('Salary')
