# SVM (Support Vector Machine).
# Import dataset.
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[, 3:5]

# Encoding the target feature as factor.
datset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

# Split dataset into training set and test set.
install.packages('caTools')

library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75) # 300 of 400 observations.
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature scaling.
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])

# Fit classifier to the training set. Put classifier here.
install.packages('e1071')
library(e1071)
classifier = svm(formula = Purchased ~ ., data = training_set, type = 'C-classification', kernel = 'linear')

# Predict test set results.
prob_pred = predict(classifier, type = 'response', newdata = test_set[-3]) # bracket removes last column.
y_pred = ifelse(prob_pred > 0.5, 1, 0) # more than 50% means they are more likely to buy.

# Make confusion matrix.
cm = table(test_set[, 3], y_pred)

# Visualize training set results.
install.packages('ElemStatLearn')

library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'Estimated Salary')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
      main = 'SVM (Training Set)',
      xlab = 'Age', ylab = 'Estimated Salary',
      xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Visualize test set results.
install.packages('ElemStatLearn')

library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'Estimated Salary')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
      main = 'SVM (Test Set)',
      xlab = 'Age', ylab = 'Estimated Salary',
      xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))
