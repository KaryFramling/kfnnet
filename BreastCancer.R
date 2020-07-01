# Tests and utilities for Breast Cancer data set. 
#
# 699 observations on 11 variables, one being a character variable, 9 being ordered or nominal, 
# and 1 target class. 
# For caret models, see https://rpubs.com/ChristianLopezB/Supervised_Machine_Learning
#
# Created by Kary Främling 4oct2019
#

require(caret)

# Get Breast Cancer data set
require(mlbench)
data(BreastCancer)

source("RBF.R")

#Shows Confusion Matrix and performance metrics
#confusionMatrix(predictions, testing.BreastCancer$Class)

kfoldcv <- trainControl(method="cv", number=10)
tc.none <- trainControl(method="none")
performance_metric <- "Accuracy"

# INKA
breastcancer.inka.run <- function(train, test) {
  exec.time <- system.time(
    inka.Breastcancer <<- train.inka.formula(Class~., data=train, spread=1, max.iter=50, 
                              classification.error.limit=0))
  nbr.hidden <- nrow(inka.Breastcancer$get.hidden()$get.weights()) # Number of hidden neurons
  # Training set performance
  y <- predict.rbf(inka.Breastcancer, newdata=train)
  classification <- (y == apply(y, 1, max)) * 1; 
  targets <- model.matrix(~0+train[,'Class']) # One-hot encoding
  train.err <- sum(abs(targets - classification)) / 2 # Simple calculation of how many mis-classified
  # Test set performance
  y <- predict.rbf(inka.Breastcancer, newdata=test)
  classification <- (y == apply(y, 1, max)) * 1; 
  targets <- model.matrix(~0+test[,'Class']) # One-hot encoding
  test.err <- sum(abs(targets - classification)) / 2 # Simple calculation of how many mis-classified
  return(data.frame("Train Set Errors"=train.err,"Test Set Errors"=test.err,"Execution time"=exec.time[3], 
                    "Size"=nbr.hidden,
                    row.names=c("inka")))
}

# Best INKA of "n".
breastcancer.best.inka.run <- function(train, test) {
  exec.time <- system.time(
    inka.best.Breastcancer <<- find.best.inka.formula(Class~., data=train, n=5, spread=1, max.iter=50, 
                                             classification.error.limit=0))
  nbr.hidden <- nrow(inka.best.Breastcancer$get.hidden()$get.weights()) # Number of hidden neurons
  # Training set performance
  y <- predict.rbf(inka.best.Breastcancer, newdata=train)
  classification <- (y == apply(y, 1, max)) * 1; 
  targets <- model.matrix(~0+train[,'Class']) # One-hot encoding
  train.err <- sum(abs(targets - classification)) / 2 # Simple calculation of how many mis-classified
  # Test set performance
  y <- predict.rbf(inka.best.Breastcancer, newdata=test)
  classification <- (y == apply(y, 1, max)) * 1; 
  targets <- model.matrix(~0+test[,'Class']) # One-hot encoding
  test.err <- sum(abs(targets - classification)) / 2 # Simple calculation of how many mis-classified
  return(data.frame("Train Set Errors"=train.err,"Test Set Errors"=test.err,"Execution time"=exec.time[3], 
                    "Size"=nbr.hidden,
                    row.names=c("inka.best(5)")))
}

#Linear Discriminant Analysis (LDA)
breastcancer.lda.run <- function(train, test) {
  exec.time <- system.time(
    lda.BreastCancer <<- train(Class~., data=train, method="lda", metric=performance_metric, trControl=kfoldcv)
  )
  
  # Test set performance
  predictions <- predict(lda.BreastCancer, newdata=test)
  test.err <- sum(predictions != test$Class)
  #print(confusionMatrix(predictions, test$Class)) # Not needed here
  # Training set performance
  predictions <- predict(lda.BreastCancer, newdata=train)
  train.err <- sum(predictions != train$Class)
  return(data.frame("Train Set Errors"=train.err,"Test Set Errors"=test.err,"Execution time"=exec.time[3], 
                    "Size"=length(lda.BreastCancer$finalModel$svd),
                    row.names=c("lda")))
}

#Classification and Regression Trees (CART)
breastcancer.cart.run <- function(train, test) {
  exec.time <- system.time(
    cart.BreastCancer <<- train(Class~., data=train, method="rpart", metric=performance_metric, trControl=kfoldcv)
  )
  
  # Test set performance
  predictions <- predict(cart.BreastCancer, newdata=test)
  test.err <- sum(predictions != test$Class)
  print(confusionMatrix(predictions, test$Class)) # Not needed here
  # Training set performance
  predictions <- predict(cart.BreastCancer, newdata=train)
  train.err <- sum(predictions != train$Class)
  return(data.frame("Train Set Errors"=train.err,"Test Set Errors"=test.err,"Execution time"=exec.time[3], 
                    "Size"=NA,
                    row.names=c("cart")))
}

#Support Vector Machines (SVM)
breastcancer.svm.run <- function(train, test) {
  exec.time <- system.time(
    svm.BreastCancer <<- train(Class~., data=train, method="svmRadial", metric=performance_metric, trControl=kfoldcv,preProcess=c("center", "scale"))
  )
  
  # Test set performance
  predictions <- predict(svm.BreastCancer, newdata=test)
  test.err <- sum(predictions != test$Class)
  print(confusionMatrix(predictions, test$Class)) # Not needed here
  # Training set performance
  predictions <- predict(svm.BreastCancer, newdata=train)
  train.err <- sum(predictions != train$Class)
  return(data.frame("Train Set Errors"=train.err,"Test Set Errors"=test.err,"Execution time"=exec.time[3], 
                    "Size"=NA,
                    row.names=c("svm")))
}

# Random Forest
breastcancer.rf.run <- function(train, test) {
  exec.time <- system.time(
    rf.BreastCancer <<- train(Class~., data=train, method="rf", metric=performance_metric, trControl=kfoldcv,preProcess=c("center", "scale"))
  )
  
  # Test set performance
  predictions <- predict(rf.BreastCancer, newdata=test)
  test.err <- sum(predictions != test$Class)
  print(confusionMatrix(predictions, test$Class)) # Not needed here
  # Training set performance
  predictions <- predict(rf.BreastCancer, newdata=train)
  train.err <- sum(predictions != train$Class)
  return(data.frame("Train Set Errors"=train.err,"Test Set Errors"=test.err,"Execution time"=exec.time[3], 
                    "Size"=rf.BreastCancer$finalModel$ntree,
                    row.names=c("rf")))
}

# Gbm
breastcancer.gbm.run <- function(train, test) {
  exec.time <- system.time(
    gbm.BreastCancer <<- train(Class~., data=train, method="gbm", metric=performance_metric, trControl=kfoldcv,preProcess=c("center", "scale"))
  )
  
  # Test set performance
  predictions <- predict(gbm.BreastCancer, newdata=test)
  test.err <- sum(predictions != test$Class)
  print(confusionMatrix(predictions, test$Class)) # Not needed here
  # Training set performance
  predictions <- predict(gbm.BreastCancer, newdata=train)
  train.err <- sum(predictions != train$Class)
  return(data.frame("Train Set Errors"=train.err,"Test Set Errors"=test.err,"Execution time"=exec.time[3], 
                    "Size"=gbm.BreastCancer$bestTune[1][1,1],
                    row.names=c("gbm")))
}

# nnet
breastcancer.nnet.run <- function(train, test) {
  exec.time <- system.time(
    nnet.BreastCancer <<- train(Class~., data=train, method="nnet", metric=performance_metric, 
                                trControl=kfoldcv, preProcess=c("center", "scale"), maxit=5000) 
  )
  
  # Test set performance
  predictions <- predict(nnet.BreastCancer, newdata=test)
  test.err <- sum(predictions != test$Class)
  print(confusionMatrix(predictions, test$Class)) # Not needed here
  # Training set performance
  predictions <- predict(nnet.BreastCancer, newdata=train)
  train.err <- sum(predictions != train$Class)
  return(data.frame("Train Set Errors"=train.err,"Test Set Errors"=test.err,"Execution time"=exec.time[3], 
                    "Size"=nnet.BreastCancer$bestTune[1][1,1],
                    row.names=c("nnet")))
}

# nnet with fixed size, found trough manual search. Strange, already with size 15 it says 
# "too many weights" (1543 or similar) and stops with error. Not much of a neural network...
breastcancer.nnet.fix.run <- function(train, test) {
  trc <- trainControl(method = "none") # Only one iteration. 
  exec.time <- system.time(
    nnet.fix.BreastCancer <<- train(Class~., data=train, method="nnet", tuneGrid = data.frame(size=10,decay=0), 
                                metric=performance_metric, 
                                trControl=trc, preProcess=c("center", "scale"), maxit=5000) 
  )
  
  # Test set performance
  predictions <- predict(nnet.fix.BreastCancer, newdata=test)
  test.err <- sum(predictions != test$Class)
  print(confusionMatrix(predictions, test$Class)) # Not needed here
  # Training set performance
  predictions <- predict(nnet.fix.BreastCancer, newdata=train)
  train.err <- sum(predictions != train$Class)
  return(data.frame("Train Set Errors"=train.err,"Test Set Errors"=test.err,"Execution time"=exec.time[3], 
                    "Size"=nnet.fix.BreastCancer$bestTune[1][1,1],
                    row.names=c("nnet.fix")))
}

# # Summary of results
# results.BreastCancer <- resamples(list(lda=lda.BreastCancer, cart=cart.BreastCancer, svm=svm.BreastCancer, rf=rf.BreastCancer))
# summary(results.BreastCancer)
# 
# # Plot results
# dotplot(results.BreastCancer)

breastcancer.run.all <- function() {
  #set.seed(7)
  
  # Remove rows with NAs. Remove ID attribute. 
  BC <- BreastCancer[complete.cases(BreastCancer),-1]
  
  # Transform all input columns to numeric, want to avoid "factor" type.
  for ( i in 1:9 ) {
    BC[,i] <- as.numeric(BC[,i])
  }
  
  # Create training and test sets
  inTrain<-createDataPartition(y=BC$Class, p=0.75, list=FALSE) # 75% to train set
  training <- BC[inTrain,]
  test <- BC[-inTrain,]
  #preObj <- preProcess(training.BreastCancer[,-11], method = c("center", "scale"))
  #preObjData <- predict(preObj,training.BreastCancer[,-11])
  #modelFit<-train(Class~., data=training.BreastCancer, method="lda")
  #Predict new data with model fitted
  #predictions<-predict(modelFit, newdata=testing.BreastCancer)
  
  # Run all models and get performance
  breastcancer.perf <- breastcancer.inka.run(training, test)
  breastcancer.perf <- rbind(breastcancer.perf, breastcancer.best.inka.run(training, test))
  breastcancer.perf <- rbind(breastcancer.perf, breastcancer.lda.run(training, test))
  breastcancer.perf <- rbind(breastcancer.perf, breastcancer.cart.run(training, test))
  breastcancer.perf <- rbind(breastcancer.perf, breastcancer.svm.run(training, test))
  breastcancer.perf <- rbind(breastcancer.perf, breastcancer.rf.run(training, test))
  breastcancer.perf <- rbind(breastcancer.perf, breastcancer.gbm.run(training, test))
  breastcancer.perf <- rbind(breastcancer.perf, breastcancer.nnet.run(training, test))
  breastcancer.perf <- rbind(breastcancer.perf, breastcancer.nnet.fix.run(training, test))
  
  return(breastcancer.perf)
}

# # CIU for BreastCancer
# BC <- BreastCancer[complete.cases(BreastCancer),-1]
# for ( i in 1:9 ) {
#   BC[,i] <- as.numeric(BC[,i])
# }
# bc_test <- BC[1,1:9] # The one to explain
# in.mins <- as.numeric(apply(BC[,1:9], 2, min))
# in.maxs <- as.numeric(apply(BC[,1:9], 2, max))
# c.minmax <- cbind(in.mins, in.maxs)
# out.names <- names(BC)[10]
# abs.min.max <- matrix(c(0,1), nrow=1)
# ciu <- ciu.new(nnet.fix.BreastCancer, in.min.max.limits=c.minmax, abs.min.max=abs.min.max, output.names=out.names)
# CI.CU <- ciu$explain(bc_test, ind.inputs.to.explain=c(1))

