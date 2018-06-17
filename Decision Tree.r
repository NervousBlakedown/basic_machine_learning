# Regression Template.
# Import the dataset.
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Split dataset into training set and test set.
# install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Salary, SplitRatio = 2/3)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature scaling.
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fit the decision tree regression model to dataset.
install.packages('rpart')
library(rpart)
regressor = rpart(formula = Salary ~ .,
                  data = dataset,
                  control = rpart.control(minsplit = 1))

# Predict new result.
y_pred = predict(regressor, data.frame(Level = 6.5))

# Same visualization as earlier, but with higher resolution and smoother curve.
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
    geom_point(aes(x = dataset$Level, y = dataset$Salary),
        color = 'red') +
    geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
        color = 'blue') +
    ggtitle('Truth or Bluff (Decision Tree Regression)') +
    xlab('Level') +
    ylab('Salary')
