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


# Predicting the test set results.
y_pred = predict(regressor, newdata = test_set)


# Building the optimal model using backward elimination.
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
                data = dataset)
summary(regressor)
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend, # 'State' proven statistically insignificant; removed.
                data = dataset)
summary(regressor)
regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend, # Removed 'Administration'.
                data = dataset)
summary(regressor)
regressor = lm(formula = Profit ~ R.D.Spend, # Removed 'marketing spend'.
                data = dataset)
summary(regressor)


# Automatic Backward Elimination.
backwardElimination <- function(x, sl) {
    numVars = length(x)
    for (i in c(1:numVars)){
      regressor = lm(formula = Profit ~ ., data = x)
      maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
      if (maxVar > sl){
        j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
        x = x[, -j]
      }
      numVars = numVars - 1
    }
    return(summary(regressor))
  }

  SL = 0.05
  dataset = dataset[, c(1,2,3,4,5)]
  backwardElimination(training_set, SL)
