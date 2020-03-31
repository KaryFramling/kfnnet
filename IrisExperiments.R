# Tests and utilities for Iris data set. 
#
# 4 input variables, three possible Iris classes. 
# For caret models, see https://rpubs.com/ChristianLopezB/Supervised_Machine_Learning
#
# Created by Kary Främling 4oct2019
#

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
  setosas <- (iris[, 5] == "setosa") * 1
  versicolors <- (iris[, 5] == "versicolor") * 1
  virginicas <- (iris[, 5] == "virginica") * 1
  targets <- matrix(c(setosas, versicolors, virginicas), ncol = 3)
  
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
iris.get.best.inka <- function(n.rbfs.to.test=1) {
  n.in <- 4
  n.out <- 3
  t.in <-
    as.matrix(iris[, 1:n.in]) # Iris data set apparently exists by default in R
  setosas <- (iris[, 5] == "setosa") * 1
  versicolors <- (iris[, 5] == "versicolor") * 1
  virginicas <- (iris[, 5] == "virginica") * 1
  targets <- matrix(c(setosas, versicolors, virginicas), ncol = 3)
  rbf <- find.best.inka (n=n.rbfs.to.test, train.inputs=t.in, train.outputs=targets, max.iter=20, 
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

iris.caret.models <- function() {
  # Create training and test sets
  inTrain<-createDataPartition(y=iris$Species, p=0.75, list=FALSE) # 75% to train set
  training.Iris<-iris[inTrain,]
  testing.Iris<-iris[-inTrain,]
  preObj<-preProcess(training.Iris[,-5], method = c("center", "scale"))
  preObjData<-predict(preObj,training.Iris[,-5])
  modelFit<-train(Species~., data=training.Iris, preProcess=c("center", "scale"), method="lda")
  #Predict new data with model fitted
  predictions<-predict(modelFit, newdata=testing.Iris)
  
  #Shows Confusion Matrix and performance metrics
  confusionMatrix(predictions, testing.Iris$Species)
  
  kfoldcv <- trainControl(method="cv", number=10)
  performance_metric <- "Accuracy"
  
  #Linear Discriminant Analysis (LDA)
  lda.iris <- train(Species~., data=iris, method="lda", metric=performance_metric, trControl=kfoldcv,preProcess=c("center", "scale"))

  #Classification and Regression Trees (CART)
  cart.iris <- train(Species~., data=iris, method="rpart", metric=performance_metric, trControl=kfoldcv,preProcess=c("center", "scale"))
  
  #Support Vector Machines (SVM)
  svm.iris <- train(Species~., data=iris, method="svmRadial", metric=performance_metric, trControl=kfoldcv,preProcess=c("center", "scale"))
  
  # Random Forest
  rf.iris <- train(Species~., data=iris, method="rf", metric=performance_metric, trControl=kfoldcv,preProcess=c("center", "scale"))
  
  # Summary of results
  results.iris <- resamples(list(lda=lda.iris, cart=cart.iris,  svm=svm.iris, rf=rf.iris))
  summary(results.iris)
  
  # Plot results
  dotplot(results.iris)
}
