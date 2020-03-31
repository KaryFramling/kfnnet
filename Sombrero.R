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

# Testing different models from caret. 
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

# Use Random Forest. 
sombrero.rf <- function() {
  #rfModel <- train(z ~ ., data=trainData, method = 'rf') # This works better
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

# Test using Neural Nets package directly
sombrero.nnet <- function() {
  preProcess_range_model <- preProcess(trainData, method='range')
  tD <- predict(preProcess_range_model, newdata = trainData)
  library(neuralnet)
  #n <- names(tD)
  f <- as.formula("z ~ .")
  nn <- neuralnet(f,data=tD,hidden=c(50),linear.output=T,stepmax=1e+06, lifesign = "full")
  pr.nn <- predict(nn,fxy)
  z <- matrix(pr.nn, nrow=length(x), ncol=length(y))
  persp(x, y, z)
  rmse.nn <<- RMSE(test.targets,pr.nn)
}

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
  
  # Calculate error measure etc.
  print(n.hidden)
  
  # Plot output surface
  zvals <- rbf$eval(plot.XYvals)
  z <- matrix(zvals, nrow=length(x), ncol=length(y))
  persp(x, y, z)
  rmse.inka <<- RMSE(test.targets,zvals)
}

