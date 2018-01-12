#Importing the dataset
dataset = read_csv('Salary_Data.csv')
#dataset = dataset, [, 2:3]

#Splitting data into training set and test set
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Feature scaling
#training_set[, 2:3] = scale(training_set[, 2:3])
#test_set[, 2:3] = scale(test_set[, 2:3])

#Fitting simple linear regression to the training set
regressor = lm(formula = Salary ~ YearsExperience, data = training_set)
