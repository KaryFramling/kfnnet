# Tests and utilities for Iris data set. 
#
# 4 input variables, three possible Iris classes. 
# For caret models, see https://rpubs.com/ChristianLopezB/Supervised_Machine_Learning
#
# Created by Kary Främling 4oct2019
#

require(caret)

source("RBF.R")
source("ContextualImportanceUtility.R")

# This DOES NOT give good results because "init.centroids.grid" covers the entire input space 
# but the training examples don't. With QR decomposition the function goes completely out of bounds 
# in the places where there are no examples. So calculating CI&CU doesn't give expected results. 
# This is not because there would be anything wrong with CI/CU but with the learning / intervals 
# for calculating CI/CU. Potential solutions:
# - Allocate kernels based on random selection from training set (or similar, 
#   INKA would probably be good choice). Would probably be good to create many RBFs and take the best one, 
#   with different kernel initialisations / order of presenting examples for INKA. 
# - Maybe somehow limit CI/CU calculation to the area that is "valid", i.e. where there are training 
#   examples present. With previously mentioned kernel allocation methods, this could be done by checking
#   somehow that there are kernels "sufficiently close". 
# - HMMM, maybe easiest could be to use grid-based allocation but then remove the kernels that are too far 
#   from any training example. dist(t(aff.trans$eval(t(t.in)))) gets distance matrix for Iris data in 
#   normalised [0,1] space.
iris.rbf.grid.test <- function() {
  n.in <- 4
  n.out <- 3
  rbf <-
    rbf.new(n.in,
            n.out,
            0,
            activation.function = squared.distance.activation,
            output.function = imqe.output.function)
  rbf$set.nrbf(TRUE)
  t.in <- as.matrix(iris[, 1:4]) # Iris data set apparently exists by default in R
  in.mins <- apply(t.in, 2, min)
  in.maxs <- apply(t.in, 2, max)

  # One-hot encoding for target variable 
  targets <- model.matrix(~0+iris[,'Species'])
  
  # Initialise hidden layer. Normalise input matrix to [0,1].
  aff.trans <-
    scale.translate.ranges(in.mins, in.maxs, c(0, 0, 0, 0), c(1, 1, 1, 1))
  nrbf <- 3 # Number of neurons per dimension, gives nrbf^4 neurons
  hl <- rbf$get.hidden()
  hl$init.centroids.grid(in.mins,
                         in.maxs,
                         c(nrbf, nrbf, nrbf, nrbf),
                         affine.transformation = aff.trans)
  ol <- rbf$get.outlayer()
  ol$set.use.bias(FALSE)
  ol$set.weights(matrix(rep(0, nrbf ^ n.in * n.out), nrow = n.out))
  rbf$set.spread(0.01)
  ty <- rbf$eval(t.in)
  hl <- rbf$get.hidden()
  h.out <- hl$get.outputs()
  w <- qr.solve(h.out, targets)
  ol$set.weights(t(w))
  y <- rbf$eval(t.in)
  classification <- (y == apply(y, 1, max)) * 1
  nbr.errors <-
    sum(abs(targets - classification)) / 2 # How many are mis-classified
  xi <- 1 # The input to plot
  yi <- 2 # Output index to plot
  interv <- (in.maxs[xi] - in.mins[xi]) / 40
  xp <- seq(in.mins[xi], in.maxs[xi], interv)
  instance.values <- c(7, 3.2, 6, 1.8)
  m <- matrix(
    instance.values,
    ncol = 4,
    nrow = length(xp),
    byrow = T
  )
  m[, xi] <- xp
  yp <- rbf$eval(m)
  plot(xp, yp[, yi], type = 'l')
  y <- rbf$eval(instance.values)
  points(instance.values[xi], y[yi], pch = 1, col = "red")
}

# Train INKA with Iris data, get best network from a number of tested ones.
iris.get.best.inka <- function(n.rbfs.to.test=1, train=NULL) {
  n.in <- 4
  n.out <- 3
  if ( is.null(train) )
    train <- iris
  t.in <-
    as.matrix(train[, 1:n.in]) # Iris data set apparently exists by default in R
  # One-hot encoding for target variable 
  targets <- model.matrix(~0+train[,'Species'])
  rbf <- find.best.inka(n=n.rbfs.to.test, train.inputs=t.in, train.outputs=targets, max.iter=30, 
                         inv.whole.set.at.end=F, classification.error.limit=0, 
                         rmse.limit=NULL, activation.function=squared.distance.activation, 
                         output.function=imqe.output.function, nrbf=T, use.bias=F, 
                         spread=0.1, c=0.01, test.inputs=NULL, test.outputs=NULL)
  
  # For getting statistics:
  # h <- rbf$get.hidden()
  # n.hidden <- nrow(h$get.weights()) # Number of hidden neurons
  # y <- rbf$eval(t.in); classification <- (y == apply(y, 1, max)) * 1; nbr.errors <- sum(abs(targets - classification)) / 2
  # print(paste("Number of hidden neurons:", n.hidden))
  # print(paste("Number of classification errors:", nbr.errors))
  return(rbf) # Return trained RBF network
}

# Train INKA with Iris data, get best network from a number of tested ones.
iris.inka.formula <- function() {
  inTrain <- createDataPartition(y=iris$Species, p=0.75, list=FALSE) # 75% to train set
  training.Iris <- iris[inTrain,]
  testing.Iris <- iris[-inTrain,]
  rbf <- train.inka.formula(Species~., data=training.Iris, spread=0.1, max.iter=20, 
                                        classification.error.limit=0)
  y <- predict.rbf(rbf, newdata=testing.Iris)
  classification <- (y == apply(y, 1, max)) * 1; 
  #perf <- sum(abs(test.out - classification)) / 2; print(perf) # Simple calculation of how many mis-classified
  # Confusion matrix. Requires converting one-hot to factor. 
  pred.class <- seq(1:nrow(classification)); for ( i in 1:3) { pred.class[classification[,i]==1] <- levels(iris$Species)[i]}
  confusionMatrix(data = as.factor(pred.class), reference = iris[-inTrain, 5])
}

iris.caret.models <- function() {
  # Create training and test sets
  inTrain<-createDataPartition(y=iris$Species, p=0.75, list=FALSE) # 75% to train set
  training.Iris<-iris[inTrain,]
  testing.Iris<-iris[-inTrain,]
  # preObj<-preProcess(training.Iris[,-5], method = c("center", "scale"))
  # preObjData<-predict(preObj,training.Iris[,-5])
  modelFit<-train(Species~., data=training.Iris, preProcess=c("center", "scale"), method="lda")
  #Predict new data with model fitted
  predictions<-predict(modelFit, newdata=testing.Iris)
  
  #Shows Confusion Matrix and performance metrics
  confusionMatrix(predictions, testing.Iris$Species)
  
  kfoldcv <- trainControl(method="cv", number=10)
  performance_metric <- "Accuracy"
  
  #Linear Discriminant Analysis (LDA)
  lda.time <- system.time(lda.iris <<- train(Species~., data=iris, method="lda", metric=performance_metric, trControl=kfoldcv,preProcess=c("center", "scale")))
  print(lda.time)
  
  #Classification and Regression Trees (CART)
  cart.iris <<- train(Species~., data=iris, method="rpart", metric=performance_metric, trControl=kfoldcv,preProcess=c("center", "scale"))
  
  #Support Vector Machines (SVM)
  svm.iris <<- train(Species~., data=iris, method="svmRadial", metric=performance_metric, trControl=kfoldcv,preProcess=c("center", "scale"))
  
  # Random Forest
  rf.iris <<- train(Species~., data=iris, method="rf", metric=performance_metric, trControl=kfoldcv,preProcess=c("center", "scale"))
  
  # Stochastic Gradient Boosting
  gbm.iris <<- train(Species~., data=iris, method="gbm", metric=performance_metric, trControl=kfoldcv,preProcess=c("center", "scale"))
  
  # nnet
  nnet.iris <<- train(Species~., data=iris, method="nnet", metric=performance_metric, trControl=kfoldcv,preProcess=c("center", "scale"))
  
  # Summary of results
  results.iris <<- resamples(list(lda=lda.iris, cart=cart.iris,  svm=svm.iris, rf=rf.iris, gbm=gbm.iris, nnet=nnet.iris))
  summary(results.iris)
  
  # Plot results
  dotplot(results.iris)
}

iris.inka.run <- function(train, test) {
  exec.time <- system.time(
    rbf <- train.inka.formula(Species~., data=train, spread=0.1, max.iter=50, 
                              classification.error.limit=0))
  nbr.hidden <- nrow(rbf$get.hidden()$get.weights()) # Number of hidden neurons
  # Training set performance
  y <- predict.rbf(rbf, newdata=train)
  classification <- (y == apply(y, 1, max)) * 1; 
  targets <- model.matrix(~0+train[,'Species']) # One-hot encoding
  train.err <- sum(abs(targets - classification)) / 2 # Simple calculation of how many mis-classified
  # Test set performance
  y <- predict.rbf(rbf, newdata=test)
  classification <- (y == apply(y, 1, max)) * 1; 
  targets <- model.matrix(~0+test[,'Species']) # One-hot encoding
  test.err <- sum(abs(targets - classification)) / 2 # Simple calculation of how many mis-classified
  return(data.frame("Train Set Errors"=train.err,"Test Set Errors"=test.err,"Execution time"=exec.time[3], 
                    "Size"=nbr.hidden,
                    row.names=c("inka")))
}

iris.best.inka.run <- function(train, test) {
#  exec.time <- system.time(rbf <- iris.get.best.inka(5)) # This needs to be modified...
  exec.time <- system.time(
    rbf <- find.best.inka.formula(Species~., data=train, n=5, spread=0.1, max.iter=30, classification.error.limit=0))
  nbr.hidden <- nrow(rbf$get.hidden()$get.weights()) # Number of hidden neurons
  # Training set performance
  #y <- rbf$eval(as.matrix(train[,1:4]))
  y <- predict.rbf(rbf, newdata=train)
  classification <- (y == apply(y, 1, max)) * 1; 
  targets <- model.matrix(~0+train[,'Species']) # One-hot encoding
  train.err <- sum(abs(targets - classification)) / 2 # Simple calculation of how many mis-classified
  # Test set performance
  #y <- rbf$eval(as.matrix(test[,1:4]))
  y <- predict.rbf(rbf, newdata=test)
  classification <- (y == apply(y, 1, max)) * 1; 
  targets <- model.matrix(~0+test[,'Species']) # One-hot encoding
  test.err <- sum(abs(targets - classification)) / 2 # Simple calculation of how many mis-classified
  return(data.frame("Train Set Errors"=train.err,"Test Set Errors"=test.err,"Execution time"=exec.time[3], 
                    "Size"=nbr.hidden,
                    row.names=c("best inka (5)")))
}

iris.lda.run <- function(train, test) {
  exec.time <- system.time(
    modelFit <- train(Species~., data=train, preProcess=c("center", "scale"), method="lda")
  )
  
  # Test set performance
  predictions <- predict(modelFit, newdata=test)
  test.err <- sum(predictions != test$Species)
  #confusionMatrix(predictions, test$Species) # Not needed here
  # Training set performance
  predictions <- predict(modelFit, newdata=train)
  train.err <- sum(predictions != train$Species)
  return(data.frame("Train Set Errors"=train.err,"Test Set Errors"=test.err,"Execution time"=exec.time[3], 
                    "Size"=length(modelFit$finalModel$svd),
                    row.names=c("lda")))
}

iris.rf.run <- function(train, test) {
  kfoldcv <- trainControl(method="cv", number=10)
  performance_metric <- "Accuracy"
  exec.time <- system.time(
    modelFit <- train(Species~., data=iris, method="rf", metric=performance_metric, trControl=kfoldcv,preProcess=c("center", "scale"))
  )
  # Test set performance
  predictions <- predict(modelFit, newdata=test)
  test.err <- sum(predictions != test$Species)
  #confusionMatrix(predictions, test$Species) # Not needed here
  # Training set performance
  predictions <- predict(modelFit, newdata=train)
  train.err <- sum(predictions != train$Species)
  return(data.frame("Train Set Errors"=train.err,"Test Set Errors"=test.err,"Execution time"=exec.time[3], 
                    "Size"=modelFit$finalModel$ntree,
                    row.names=c("rf")))
}

iris.gbm.run <- function(train, test) {
  kfoldcv <- trainControl(method="cv", number=10)
  performance_metric <- "Accuracy"
  exec.time <- system.time(
    modelFit <- train(Species~., data=iris, method="gbm", metric=performance_metric, trControl=kfoldcv,preProcess=c("center", "scale"))
  )
  # Test set performance
  predictions <- predict(modelFit, newdata=test)
  test.err <- sum(predictions != test$Species)
  #confusionMatrix(predictions, test$Species) # Not needed here
  # Training set performance
  predictions <- predict(modelFit, newdata=train)
  train.err <- sum(predictions != train$Species)
  return(data.frame("Train Set Errors"=train.err,"Test Set Errors"=test.err,"Execution time"=exec.time[3], 
                    "Size"=modelFit$bestTune[1][1,1],
                    row.names=c("gbm")))
}

iris.nnet.run <- function(train, test) {
  kfoldcv <- trainControl(method="cv", number=10)
  performance_metric <- "Accuracy"
  exec.time <- system.time(
    # We want to run until it converges
    nnet.iris <<- train(Species~., data=train, method="nnet", metric=performance_metric, 
                      trControl=kfoldcv, preProcess=c("center", "scale"), maxit=5000) 
  )
  
  # Test set performance
  predictions <- predict(nnet.iris, newdata=test)
  test.err <- sum(predictions != test$Species)
  #confusionMatrix(predictions, test$Species) # Not needed here
  # Training set performance
  predictions <- predict(nnet.iris, newdata=train)
  train.err <- sum(predictions != train$Species)
  return(data.frame("Train Set Errors"=train.err,"Test Set Errors"=test.err,"Execution time"=exec.time[3], 
                    "Size"=nnet.iris$bestTune[1][1,1],
                    row.names=c("nnet")))
}

iris.nnet.fix.run <- function(train, test) {
  trc <- trainControl(method = "none") # Only one iteration. 
  performance_metric <- "Accuracy"
  exec.time <- system.time(
    # We want to run until it converges
    nnet.fix.iris <<- train(Species~., data=train, method="nnet", tuneGrid = data.frame(size=15,decay=0), 
                        metric=performance_metric, 
                        trControl=trc, preProcess=c("center", "scale"), maxit=5000) 
  )
  
  # Test set performance
  predictions <- predict(nnet.fix.iris, newdata=test)
  test.err <- sum(predictions != test$Species)
  #confusionMatrix(predictions, test$Species) # Not needed here
  # Training set performance
  predictions <- predict(nnet.fix.iris, newdata=train)
  train.err <- sum(predictions != train$Species)
  return(data.frame("Train Set Errors"=train.err,"Test Set Errors"=test.err,"Execution time"=exec.time[3], 
                    "Size"=nnet.fix.iris$bestTune[1][1,1],
                    row.names=c("nnet.fix")))
}

iris.run.all <- function() {
  #set.seed(7)
  
  # Create training and test sets
  inTrain <- createDataPartition(y=iris$Species, p=0.75, list=FALSE) # 75% to train set
  training <- iris[inTrain,]
  test <- iris[-inTrain,]
  
  # Run all models and get performance
  iris.perf <- iris.inka.run(training, test)
  iris.perf <- rbind(iris.perf, iris.best.inka.run(training, test))
  iris.perf <- rbind(iris.perf, iris.lda.run(training, test))
  iris.perf <- rbind(iris.perf, iris.rf.run(training, test))
  iris.perf <- rbind(iris.perf, iris.gbm.run(training, test))
  iris.perf <- rbind(iris.perf, iris.nnet.run(training, test))
  iris.perf <- rbind(iris.perf, iris.nnet.fix.run(training, test))

  return(iris.perf)
}


