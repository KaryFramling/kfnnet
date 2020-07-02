# Experiments with "sombrero" function.
library(caret)
source("rbf.R")

# Return output value for the "sombrero" function in its 3D version,
# i.e. 2 inputs, one output. The input values may be scalars, vectors,
# matrices etc.
get.sombrero.3D <- function(x, y) {
  tmp <- sqrt(x^2 + y^2)
  return (sin(tmp)/tmp)
}

# Get "n.samples" random (x,y) pairs from given intervals, calculate corresponding
# z values for sombrero function and return (x,y,z) matrix with n.samples rows. 
# Example: m <- get.sombrero.samples(xrange=c(-10,10), yrange=c(-10,10), n.samples=100)
get.sombrero.samples <- function(xrange=c(-10,10), yrange=c(-10,10), n.samples=100) {
  x <- runif(n.samples, min=xrange[1], max=xrange[2])
  y <- runif(n.samples, min=yrange[1], max=yrange[2])
  z <- get.sombrero.3D(x,y)
  return(matrix(c(x,y,z), nrow=n.samples, byrow=FALSE))
}

# To get a 3D-plot, execute the following.
plot.sombrero.3D <- function(xrange=c(-10,10), yrange=c(-10,10), xystep=0.21, ...) {
  xmin <- xrange[1]
  xmax <- xrange[2]
  ymin <- yrange[1]
  ymax <- yrange[2]
  xstep <- xystep[1]
  if ( length(xystep) > 1 )
    ystep <- xystep[2]
  else
    ystep <- xstep
  xseq <- seq(xmin, xmax, xstep)
  yseq <- seq(ymin, ymax, ystep)
  xvals <- rep(xseq, length(yseq))
  yvals <- rep(yseq, each=length(xseq))
  zvals <- get.sombrero.3D(xvals, yvals)
  z <- matrix(zvals, nrow=length(xseq), ncol=length(yseq))
  persp(xseq, yseq, z, ...)
}

# Create training set. No noise. 
sombrero.data <- get.sombrero.samples(xrange=c(-10,10), yrange=c(-10,10), n.samples=300)
trainData <- as.data.frame(sombrero.data)
names(trainData) <- c( "x", "y", "z")

# Create test&plot set.
xmin <- ymin <- -10
xmax <- ymax <- 10
xstep <- ystep <- 0.6 # Have to avoid (0,0) because it gives NaN for sombrero

# Test & plotting set
x <- seq(xmin,xmax,xstep)
y <- seq(ymin,ymax,ystep)
l <- list(y,x)
plot.XYvals <- create.permutation.matrix(l)
fxy <- as.data.frame(plot.XYvals); names(fxy) <- c("x", "y")
test.targets <- get.sombrero.3D(plot.XYvals[,1],plot.XYvals[,2])

# Do this if you want plot of function. 
#z <- matrix(test.targets, nrow=length(x), ncol=length(y))
#persp(x, y, z)


# k Nearest Neighbors. Super-quick (no learning, actually...) and gives sombrero-looking result, 
# even though it's just about as rough as the one by Random Forest. 
sombrero.knn <- function() {
  logRegModel <- train(z ~ ., data=trainData, method = 'knn') # Works
  #logRegModel <- train(z ~ ., data=trainData, method = 'mlp', size=100) # Not good
  #logRegModel <- train(z ~ ., data=trainData, preProc = c("center", "scale"), method = 'mlpKerasDecay') # Doesn't work
  #logRegModel <- train(z ~ ., data=trainData, preProc = c("center", "scale"), method = 'nnet') # Not good
  #logRegPrediction <- predict(logRegModel, testData)
  zvals <- predict(logRegModel, fxy)
  z <- matrix(zvals, nrow=length(x), ncol=length(y))
  persp(x, y, z)
  #postResample(pred = test_set$pred, obs = test_set$obs)
  rmse.knn <<- RMSE(test.targets,zvals)
}

# Random Forest. Quick and gets to "something" that looks just a little like a sombrero.
sombrero.rf <- function() {
  rfModel <- train(z ~ ., data=trainData, method = 'rf') # This works better
   # library(randomForest)
   # rfModel <- randomForest(z ~ .,
   #                         data=trainData,
   #                         importance=TRUE,
   #                         ntree=5000)
  zvals <- predict(rfModel, fxy)
  z <- matrix(zvals, nrow=length(x), ncol=length(y))
  persp(x, y, z)
  #postResample(pred = test_set$pred, obs = test_set$obs)
  rmse.rf <<- RMSE(test.targets,zvals)
}

# Use Extreme Gradient Boosting, v1. Takes time and fails completely. 
sombrero.xgbDART <- function() {
  xgbDartModel <- train(z ~ ., data=trainData, method = 'xgbDART') 
  zvals <- predict(xgbDartModel, fxy)
  z <- matrix(zvals, nrow=length(x), ncol=length(y))
  persp(x, y, z)
  #postResample(pred = test_set$pred, obs = test_set$obs)
  rmse.xgbDART <<- RMSE(test.targets,zvals)
}

# Use Extreme Gradient Boosting, Tree version. Longer than GBM but fails in same way. 
sombrero.xgbTree <- function() {
  xgbTreeModel <- train(z ~ ., data=trainData, method = 'xgbTree') 
  zvals <- predict(xgbTreeModel, fxy)
  z <- matrix(zvals, nrow=length(x), ncol=length(y))
  persp(x, y, z)
  #postResample(pred = test_set$pred, obs = test_set$obs)
  rmse.xgbTree <<- RMSE(test.targets,zvals)
}

# Stochastic Gradient Boosting. Quick but fails completely.
sombrero.gbm <- function() {
  gbmModel <- train(z ~ ., data=trainData, method = 'gbm') 
  zvals <- predict(gbmModel, fxy)
  z <- matrix(zvals, nrow=length(x), ncol=length(y))
  persp(x, y, z)
  #postResample(pred = test_set$pred, obs = test_set$obs)
  rmse.gbm <<- RMSE(test.targets,zvals)
}

# Multilayer Perceptron Network with Dropout. Takes very long and only comes up with a linear 
# model that doesn't make sense. 
sombrero.mlpKerasDropout <- function() {
  mlpKerasDropoutModel <- train(z ~ ., data=trainData, method = 'mlpKerasDropout', linout = TRUE) 
  zvals <- predict(mlpKerasDropoutModel, fxy)
  z <- matrix(zvals, nrow=length(x), ncol=length(y))
  persp(x, y, z)
  #postResample(pred = test_set$pred, obs = test_set$obs)
  rmse.mlpKerasDropout <<- RMSE(test.targets,zvals)
}

# nnet.
sombrero.nnet <- function() {
  trc <- trainControl(method = "none") # Seems to work fine like this also! And does only one iteration. 
  start.time <- proc.time()        # Save starting time
  nnetModel <<- train(z ~ ., data=trainData,
                     method = "nnet", tuneGrid = data.frame(size=110,decay=0), trControl=trc,
                     linout = TRUE, maxit=5000)
  end.time <- proc.time()
  exec.time <- end.time - start.time # Time difference between start & end
  print(exec.time) 
  zvals <- predict(nnetModel, fxy)
  z <- matrix(zvals, nrow=length(x), ncol=length(y))
  persp(x, y, z)
  #postResample(pred = test_set$pred, obs = test_set$obs)
  rmse.nnet <<- RMSE(test.targets,zvals)
}


# # Test using Neural Nets package directly. Doesn't work...
# sombrero.nnet <- function() {
#   preProcess_range_model <- preProcess(trainData, method='range')
#   tD <- predict(preProcess_range_model, newdata = trainData)
#   library(nnet)
#   #n <- names(tD)
#   f <- as.formula("z ~ .")
#   nn <- nnet(f,data=tD,size=50,linout=TRUE,decay=0,maxit=10000)
#   pr.nn <- predict(nn,fxy)
#   z <- matrix(pr.nn, nrow=length(x), ncol=length(y))
#   persp(x, y, z)
#   rmse.nn <<- RMSE(test.targets,pr.nn)
# }

# Make INKA training on sombrero function data and plot the resulting model.
sombrero.inka.test <- function() {
  m <- sombrero.data
  t.in <-m[,1:2]
  targets <- m[,3]
  n.in <- ncol(t.in)
  n.out <- 1
  rbf <-
    rbf.new(n.in,
            n.out,
            0,
            activation.function = squared.distance.activation,
            output.function = imqe.output.function)
  rbf$set.nrbf(TRUE)
  ol <- rbf$get.outlayer()
  ol$set.use.bias(FALSE)
  rbf$set.spread(30) # d^2 parameter in INKA
  c <- 1 # The "c" parameter in INKA training, minimal distance for adding new hidden neuron.
  start.time <- proc.time()        # Save starting time
  n.hidden <-
    train.inka(
      rbf,
      t.in,
      targets,
      c,
      max.iter = 200,
      inv.whole.set.at.end = F, 
      rmse.limit=0.001
    )
  end.time <- proc.time()
  exec.time <- end.time - start.time # Time difference between start & end
  print(exec.time) 
  
  # Calculate error measure etc.
  print(paste("Number of hidden neurons:", n.hidden))
  
  # Plot output surface
  zvals <- rbf$eval(plot.XYvals)
  z <- matrix(zvals, nrow=length(x), ncol=length(y))
  persp(x, y, z)
  rmse.inka <<- RMSE(test.targets,zvals)
}


# INKA 
sombrero.inka.run <- function(train, test) {
  exec.time <- system.time(
    sombrero.rbf <<- train.inka.formula(z~., data=train, spread=30, c=1, max.iter=200, 
                              rmse.limit=0.001))
  nbr.hidden <- nrow(sombrero.rbf$get.hidden()$get.weights()) # Number of hidden neurons
  # Training set performance
  zvals <- predict.rbf(sombrero.rbf, newdata=train)
  train.err <- RMSE(train$z,zvals)
  # Test set performance
  zvals <- predict.rbf(sombrero.rbf, newdata=test)
  test.err <- RMSE(test$z,zvals)
  return(data.frame("Train Set RMSE"=train.err,"Test Set RMSE"=test.err,"Execution time"=exec.time[3], 
                    "Size"=nbr.hidden,
                    row.names=c("inka")))
}

# k-NearestNeighbor 
sombrero.knn.run <- function(train, test) {
  exec.time <- system.time(
    sombrero.knn <<- train(z ~ ., data=train, method = 'knn'))
  # Training set performance
  zvals <- predict(sombrero.knn, newdata=train)
  train.err <- RMSE(train$z,zvals)
  # Test set performance
  zvals <- predict(sombrero.knn, newdata=test)
  test.err <- RMSE(test$z,zvals)
  return(data.frame("Train Set RMSE"=train.err, "Test Set RMSE"=test.err, "Execution time"=exec.time[3], 
                    "Size"=NA,
                    row.names=c("knn")))
}

# Random Forest 
sombrero.rf.run <- function(train, test) {
  exec.time <- system.time(
    sombrero.rf <<- train(z ~ ., data=train, method = 'rf'))
  # Training set performance
  zvals <- predict(sombrero.rf, newdata=train)
  train.err <- RMSE(train$z,zvals)
  # Test set performance
  zvals <- predict(sombrero.rf, newdata=test)
  test.err <- RMSE(test$z,zvals)
  return(data.frame("Train Set RMSE"=train.err, "Test Set RMSE"=test.err, "Execution time"=exec.time[3], 
                    "Size"=sombrero.rf$finalModel$ntree,
                    row.names=c("rf")))
}

# Caret size optimized nnet 
sombrero.nnet.run <- function(train, test) {
  kfoldcv <- trainControl(method="cv", number=10)
  exec.time <- system.time(
    sombrero.nnet <<- train(z ~ ., data=train,
                                method = "nnet", trControl=kfoldcv,
                                linout = TRUE, maxit=5000))
  # Training set performance
  zvals <- predict(sombrero.nnet, newdata=train)
  train.err <- RMSE(train$z,zvals)
  # Test set performance
  zvals <- predict(sombrero.nnet, newdata=test)
  test.err <- RMSE(test$z,zvals)
  return(data.frame("Train Set RMSE"=train.err, "Test Set RMSE"=test.err, "Execution time"=exec.time[3], 
                    "Size"=sombrero.nnet$bestTune[1][1,1],
                    row.names=c("nnet")))
}

# Fixed size nnet 
sombrero.nnet.fixed.run <- function(train, test) {
  trc <- trainControl(method = "none")
  exec.time <- system.time(
    sombrero.nnet.fix <<- train(z ~ ., data=train,
                        method = "nnet", tuneGrid = data.frame(size=100,decay=0), trControl=trc,
                        linout = TRUE, maxit=5000))
  # Training set performance
  zvals <- predict(sombrero.nnet.fix, newdata=train)
  train.err <- RMSE(train$z,zvals)
  # Test set performance
  zvals <- predict(sombrero.nnet.fix, newdata=test)
  test.err <- RMSE(test$z,zvals)
  return(data.frame("Train Set RMSE"=train.err, "Test Set RMSE"=test.err, "Execution time"=exec.time[3], 
                    "Size"=sombrero.nnet.fix$bestTune[1][1,1],
                    row.names=c("nnet.fix")))
}

sombrero.run.all <- function(plot=FALSE) {
  
  # Create training set. No noise. 
  xmin <- ymin <- -10
  xmax <- ymax <- 10
  sombrero.data <- get.sombrero.samples(xrange=c(xmin,xmax), yrange=c(ymin,xmax), n.samples=300)
  training <- as.data.frame(sombrero.data)
  names(training) <- c( "x", "y", "z")
  
  # Create test&plot set.
  xstep <- ystep <- 0.6 # Have to avoid (0,0) because it gives NaN for sombrero
  x <- seq(xmin,xmax,xstep)
  y <- seq(ymin,ymax,ystep)
  l <- list(y,x)
  plot.XYvals <- create.permutation.matrix(l)
  fxy <- as.data.frame(plot.XYvals); names(fxy) <- c("x", "y")
  z <- get.sombrero.3D(plot.XYvals[,1],plot.XYvals[,2])
  test <- cbind(fxy, z)

  # Run all models and get performance
  sombrero.perf <- sombrero.inka.run(training, test)
  sombrero.perf <- rbind(sombrero.perf, sombrero.knn.run(training, test))
  sombrero.perf <- rbind(sombrero.perf, sombrero.rf.run(training, test))
  sombrero.perf <- rbind(sombrero.perf, sombrero.nnet.fixed.run(training, test))
  sombrero.perf <- rbind(sombrero.perf, sombrero.nnet.run(training, test))
  
  if ( plot ) {
    z <- matrix(zvals <- predict.rbf(sombrero.rbf, newdata=test), nrow=length(x), ncol=length(y)); persp(x, y, z, main="INKA")
    z <- matrix(zvals <- predict(sombrero.knn, newdata=test), nrow=length(x), ncol=length(y)); persp(x, y, z, main="knn")
    z <- matrix(zvals <- predict(sombrero.rf, newdata=test), nrow=length(x), ncol=length(y)); persp(x, y, z, main="Random Forest")
    z <- matrix(zvals <- predict(sombrero.nnet.fix, newdata=test), nrow=length(x), ncol=length(y)); persp(x, y, z, main="Fixed nnet")
    z <- matrix(zvals <- predict(sombrero.nnet, newdata=test), nrow=length(x), ncol=length(y)); persp(x, y, z, main="nnet")
  }

  return(sombrero.perf)
}

