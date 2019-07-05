#SVM
# Importing the dataset
dataset = read.csv("C:/Users/Karen Leguizamo/Customer-Churn-Prediction-Model/dataset.csv")
dataset = dataset[4:14]
View(dataset)
# Encoding the categorical variables as factors
dataset$Geography = as.numeric(factor(dataset$Geography,
                                      levels = c('France', 'Spain', 'Germany'),
                                      labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                                   levels = c('Female', 'Male'),
                                   labels = c(1, 2)))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
View(split)

summary(split)

training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-11] = scale(training_set[-11])### nO sabemos que hace exactamente
View(training_set[-11])
test_set[-11] = scale(test_set[-11])

# Applying PCA
# install.packages('caret')
library(caret)
# install.packages('e1071')
library(e1071)
pca = preProcess(x = training_set[-11], method = 'pca', pcaComp = 10)
training_set = predict(pca, training_set)

pca1 = preProcess(x = test_set[-11], method = 'pca', pcaComp = 10)###esto no estaba
test_set = predict(pca1, test_set)
summary(pca1)



# Fitting Kernel SVM to the Training set
# install.packages('e1071')
library(e1071)
classifier = svm(formula = Exited ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'radial')

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-1])

# Applying k-Fold Cross Validation
# install.packages('caret')
library(caret)
folds = createFolds(training_set$Exited, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier = svm(formula = Exited ~ .,
                   data = training_fold,
                   type = 'C-classification',
                   kernel = 'radial')
  y_pred = predict(classifier, newdata = test_fold[-1])
  cm = table(test_fold[, 1], y_pred)
  accuracy_svm = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy_svm)
})
accuracy_svm = mean(as.numeric(cv))

# ANN
# Importing the dataset
dataset = read.csv("C:/Users/Karen Leguizamo/Customer-Churn-Prediction-Model/dataset.csv")
dataset = dataset[4:14]

# Encoding the categorical variables as factors
dataset$Geography = as.numeric(factor(dataset$Geography,
                                      levels = c('France', 'Spain', 'Germany'),
                                      labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                                   levels = c('Female', 'Male'),
                                   labels = c(1, 2)))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-11] = scale(training_set[-11])
test_set[-11] = scale(test_set[-11])

# Applying PCA
# install.packages('caret')
library(caret)
# install.packages('e1071')
library(e1071)
pca = preProcess(x = training_set[-11], method = 'pca', pcaComp = 10)
training_set = predict(pca, training_set)

test_set = predict(pca, test_set)

# Fitting ANN to the Training set
# install.packages('h2o')
library(h2o)
h2o.init(nthreads = -1)
model = h2o.deeplearning(y = 'Exited',
                         training_frame = as.h2o(training_set),
                         activation = 'Rectifier',
                         hidden = c(5,5),
                         epochs = 100,
                         train_samples_per_iteration = -2)

# Predicting the Test set results
y_pred = h2o.predict(model, newdata = as.h2o(test_set[-1]))
y_pred = (y_pred > 0.5)
y_pred = as.vector(y_pred)


# Applying k-Fold Cross Validation
# install.packages('caret')
library(caret)
folds = createFolds(training_set$Exited, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  model = h2o.deeplearning(y = 'Exited',
                           training_frame = as.h2o(training_fold),
                           activation = 'Rectifier',
                           hidden = c(5,5),
                           epochs = 100,
                           train_samples_per_iteration = -2)
  y_pred = h2o.predict(model, newdata = as.h2o(test_fold[-1]))
  y_pred = (y_pred > 0.5)
  y_pred = as.vector(y_pred)
  cm = table(test_fold[, 1], y_pred)
  accuracy_ann = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy_ann)
})
accuracy_ann = mean(as.numeric(cv))


# XGBoost
# Importing the dataset
dataset = read.csv("C:/Users/Karen Leguizamo/Customer-Churn-Prediction-Model/dataset.csv")
dataset = dataset[4:14]

# Encoding the categorical variables as factors
dataset$Geography = as.numeric(factor(dataset$Geography,
                                      levels = c('France', 'Spain', 'Germany'),
                                      labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                                   levels = c('Female', 'Male'),
                                   labels = c(1, 2)))



# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# Fitting XGBoost to the Training set
# install.packages('xgboost')
library(xgboost)
classifier = xgboost(data = as.matrix(training_set[-11]), label = training_set$Exited, nrounds = 11)

# Predicting the Test set results
y_pred = predict(classifier, newdata = as.matrix(test_set[-11]))
y_pred = (y_pred >= 0.5)


# Applying k-Fold Cross Validation
# install.packages('caret')
library(caret)
folds = createFolds(training_set$Exited, k = 10)
cv = lapply(folds, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier = xgboost(data = as.matrix(training_set[-11]), label = training_set$Exited, nrounds = 10)
  y_pred = predict(classifier, newdata = as.matrix(test_fold[-11]))
  y_pred = (y_pred >= 0.5)
  cm = table(test_fold[, 11], y_pred)
  accuracy_xgb = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy_xgb)
})
accuracy_xgb = mean(as.numeric(cv))
