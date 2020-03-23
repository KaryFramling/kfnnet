source("Functions.R")
source("Interfaces.R")
source("NeuralLayer.R")
source("Adaline.R")
source("RBF.R")

# Calculate Contextual Importance (CI) and Contextual Utility (CU) for the "black-box" bb.
# "bb" must be an object that supports "eval" method and that only takes a vector/matrix as input
# Inputs: "black-box" object, input vector, vector of input indices, mx2 matrix of min-max values of outputs
# "montecarlo.samples" is the number of random values to use for estimating CI and CU.
# Returns: mx2 matrix with CI, CU for all outputs
# Might be useful also to return the estimated minimal and maximal values found. Would then be mx2 matrix again
contextual.IU<-function(bb, inputs, c.indices, minmax.outputs, montecarlo.samples=100, c.minmax=NULL) {
  # Create matrix of inputs using the provided values, replacing the indicated columns with random values.
  n<-montecarlo.samples
  mcm <- matrix(inputs,ncol=length(inputs),nrow=n,byrow=TRUE)
  nbr.mc.cols <- length(c.indices)
  # Special treatment for [0,1] values, makes it more efficient. 
  if ( is.null(c.minmax) ) {
    mcm[,c.indices] <- matrix(runif(n*nbr.mc.cols), nrow=n)
  }
  else {
    # Different treatment required if various min-max ranges for inputs (not [0,1]). NOT IMPLEMENTED!
    mins <- c.minmax[c.indices,1]
    diffs <- c.minmax[c.indices,2] - mins
    mcm[,c.indices] <- matrix(mins, nrow=n, ncol=nbr.mc.cols, byrow=T) + 
      matrix(runif(n*nbr.mc.cols), nrow=n)*matrix(diffs, nrow=n, ncol=nbr.mc.cols, byrow=T)
  }
  
  # Evaluate all output values
  mcout <- bb$eval(mcm)
  minvals <- apply(mcout,2,min)
  range <- apply(mcout,2,max) - minvals
  output_ranges <- matrix(minmax.outputs[,2] - minmax.outputs[,1], ncol=1)
  CI <- range/output_ranges
  
  # Calculate CU.
  cu.val <- bb$eval(matrix(inputs, nrow=1))
  CU <- (cu.val - minvals)/range
  return(matrix(c(CI,CU), ncol=2))
}

# Function for plotting out the effect of changing values of one input on one output
# bb:
# etc:
plot.CI.CU <- function(bb, instance.values, ind.input, ind.output, in.mins, in.maxs, n.points=40, xlab="x", ylab="y", ...) {
  interv <- (in.maxs[ind.input] - in.mins[ind.input])/n.points
  xp <- seq(in.mins[ind.input],in.maxs[ind.input],interv)
  if ( is.matrix(instance.values) ) 
    n.col <- ncol(instance.values)
  else
    n.col <- length(instance.values)
  m <- matrix(instance.values, ncol=n.col, nrow=length(xp), byrow=T)
  m[,ind.input] <- xp
  yp <- bb$eval(m)
  plot(xp, yp[,ind.output], type='l', xlab=xlab, ylab=ylab, ...)
  y <- bb$eval(instance.values)
  points(instance.values[ind.input], y[ind.output], col = "red", pch = 16, cex = 2)
}

# Function for 3D plotting the effect of changing values of two inputs on one output
# bb:
# ...
# n.points: How many x/y values for the plot between in.mins and in.maxs.
# etc:
plot.CI.CU.3D <- function(bb, instance.values, ind.inputs, ind.output, in.mins, in.maxs, n.points=40, ...) {
  interv <- (in.maxs[ind.inputs] - in.mins[ind.inputs])/n.points
  xp <- seq(in.mins[ind.inputs[1]], in.maxs[ind.inputs[1]], by=interv[1])
  yp <- seq(in.mins[ind.inputs[2]], in.maxs[ind.inputs[2]], by=interv[2])
  l <- list(xp,yp)
  pm <- create.permutation.matrix(l)
  if ( is.matrix(instance.values) ) 
    n.col <- ncol(instance.values)
  else
    n.col <- length(instance.values)
  m <- matrix(instance.values, ncol=n.col, nrow=length(xp)*length(yp), byrow=T)
  m[,ind.inputs[1]] <- pm[,1]
  m[,ind.inputs[2]] <- pm[,2]
  z <- bb$eval(m)
  zm <- matrix(z[,ind.output], nrow = length(xp), byrow = TRUE)
  vt <- persp(xp, yp, zm, ticktype = "detailed", ...) # persp3D might want these: , bg="white", colvar=NULL, col="black", facets=FALSE
  x.plot <- instance.values[ind.inputs[1]]
  y.plot <- instance.values[ind.inputs[2]]
  z.plot <- bb$eval(matrix(c(x.plot,y.plot), nrow=1))
  points(trans3d(x.plot, y.plot, z.plot[ind.output], pmat = vt), col = "red", pch = 16, cex = 3)
}

# Call e.g. "adaline.three.inputs.test()".
# Or "adaline.three.inputs.test(indices=c(1,3))" for getting joint importance of inputs one and three. 
adaline.three.inputs.test <- function(inp=c(0.1,0.2,0.3), indices=c(1), n.samples=100) {
  a <- adaline.new(3, 1)
  inp <- c(0.1,0.2,0.3)
  w <- c(0.20,0.30,0.50)
  a$set.weights(matrix(w, nrow=1, byrow=T))
  CI.CU <- contextual.IU(a, inp, indices, matrix(c(0, 1), nrow=1, byrow=T), n.samples)
  CI.CU
}

## Two outputs
# Call e.g. "adaline.two.outputs.test()"
# Or "aadaline.two.outputs.test(indices=c(1,3))" for getting joint importance of inputs one and three. 
adaline.two.outputs.test <- function(inp=c(0.1,0.2,0.3), indices=c(1), n.samples=100) {
  a <- adaline.new(3, 2)
  w <- matrix(c(0.20,0.30,0.50,0.25,0.35,0.40), nrow=2, byrow=TRUE)
  a$set.weights(w)
  #out2 <- a2$eval(inp2)
  CI.CU <- contextual.IU(a, inp, indices, matrix(c(0,1,0,1), nrow=2, byrow=T), montecarlo.samples=n.samples)
  CI.CU
}


# Matrix inversion for learning output layer weights
# y = Ax (slight confusion with variable naming in R documentation in my opinion)
# Two inputs, four data points (overdetermined). With Bias term.
# x <- matrix(c(
#   0,0,1,
#   5,5,1,
#   5,0,1,
#   0,5,1
# ), nrow=4, byrow=T)
# y <- matrix(c(0.1,1,0.55,0.55))
# H <- qr.solve(x, y)
# x%*%H

# Create small training set and train with QR-decomposition
rbf.regression.test <- function(indices=c(1)) {
  x <- matrix(c(
    0,0,
    1,1,
    1,0,
    0,1
  ), ncol=2, byrow=T)
  y <- matrix(c(0,1,0.3,0.6),ncol=1)
  xints <- 2
  yints <- 2
  n.hidden <- xints*yints
  rbf <- rbf.new(2, 1, n.hidden, activation.function=squared.distance.activation, 
                 output.function=imqe.output.function)
  hl <- rbf.classifier.new(nbrInputs=2, nbrOutputs=n.hidden, activation.function=squared.distance.activation, 
                           output.function=gaussian.output.function)
  mins <- c(0,0)
  maxs <- c(1,1)
  at <- scale.translate.ranges(mins, maxs, c(0,0), c(1,1))
  #hl$init.centroids.grid(mins, maxs, c(xints,yints),affine.transformation=at)
  hl$init.centroids.grid(mins, maxs, c(xints,yints))
  rbf$set.hidden(hl)
  rbf$set.nrbf(TRUE)
  rbf$set.spread(0.3)
  outp <- rbf$eval(x)
  h.out <- hl$get.outputs()
  w <- qr.solve(h.out, y)
  ol <- rbf$get.outlayer()
  ol$set.weights(t(w))
  xp <- seq(0,1,0.05)
  yp <- xp
  m<-create.input.matrix(c(1,2), c(0,0), c(1,1), c(0.05,0.05),0)
  z <- rbf$eval(m)
  zm <- matrix(data=z, nrow=length(xp), ncol=length(yp), byrow=TRUE)
  persp(xp, yp, zm, zlim=c(0,1), theta=15, phi=20, shade=0.3)
  CI.CU <- contextual.IU(rbf, matrix(c(0.5,0.1),ncol=2), indices, matrix(c(0,1), ncol=2, byrow=T))
  CI.CU
}
# rbf.regression.test()

# Create small training set for 2-class classification, train with QR-decomposition
rbf.classification.test <- function(indices=c(1), visualize.output.index=1, n.samples=100) {
  x <- matrix(c(
    0,0,
    1,1,
    1,0,
    0,1
  ), ncol=2, byrow=T)
  y <- matrix(c(
    1,0,
    0,1,
    1,0,
    1,0),ncol=2,byrow=T)
  xints <- 2
  yints <- 2
  n.hidden <- xints*yints
  rbf <- rbf.new(2, 2, n.hidden, activation.function=squared.distance.activation, 
                 output.function=imqe.output.function)
  hl <- rbf.classifier.new(nbrInputs=2, nbrOutputs=n.hidden, activation.function=squared.distance.activation, 
                           output.function=gaussian.output.function)
  mins <- c(0,0)
  maxs <- c(1,1)
  at <- scale.translate.ranges(mins, maxs, c(0,0), c(1,1))
  #hl$init.centroids.grid(mins, maxs, c(xints,yints),affine.transformation=at)
  hl$init.centroids.grid(mins, maxs, c(xints,yints))
  rbf$set.hidden(hl)
  rbf$set.nrbf(TRUE)
  rbf$set.spread(0.3)
  outp <- rbf$eval(x)
  h.out <- hl$get.outputs()
  w <- qr.solve(h.out, y)
  ol <- rbf$get.outlayer()
  ol$set.weights(t(w))
  xp <- seq(0,1,0.05)
  yp <- xp
  m<-create.input.matrix(c(1,2), c(0,0), c(1,1), c(0.05,0.05),0)
  z <- rbf$eval(m)
  zm <- matrix(data=z[,visualize.output.index], nrow=length(xp), ncol=length(yp), byrow=TRUE)
  res <- persp(xp, yp, zm, xlab = "x", ylab = "y", zlab = "z", theta = 15, phi = 5, ticktype = "detailed", zlim=c(0,1), shade=0.3)
  #round(res, 3)
  #xE <- c(0,1); xy <- expand.grid(xE, xE)
  xy.ci.cu <- matrix(c(0.5,0.1),ncol=2)
  z.ci.cu <- rbf$eval(xy.ci.cu)
  points(trans3d(xy.ci.cu[,1], xy.ci.cu[,2], z.ci.cu[,visualize.output.index], pmat = res), col = 2, pch = 16, cex = 3)
  CI.CU <- contextual.IU(rbf, xy.ci.cu, indices, matrix(c(0,1,0,1), ncol=2, byrow=T), montecarlo.samples=n.samples)
  CI.CU
}
#rbf.classification.test()

