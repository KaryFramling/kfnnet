# "R" implementation of Contextual Importance and Utility.
#
# Kary Främling, created in 2019
#

source("Interfaces.R")
source("Functions.R")

# Create CIU object
# Calculate Contextual Importance (CI) and Contextual Utility (CU) for the "black-box" bb.
# "bb" must be an object that supports "eval" method and that only takes a vector/matrix as input
# Inputs: "black-box" object, input vector, vector of input indices, mx2 matrix of min-max values of outputs
# "montecarlo.samples" is the number of random values to use for estimating CI and CU.
# Returns: mx2 matrix with CI, CU for all outputs
# Might be useful also to return the estimated minimal and maximal values found. Would then be mx2 matrix again
ciu.new <- function(bb, train.inputs=NULL, train.targets=NULL, abs.min.max=NULL, output.names=NULL) {
  model <- bb
  t.in <- train.inputs
  t.target <- train.targets
  minmax.outputs <- abs.min.max
  ciu.row.names <- output.names
  n.mc.samples <- NULL 
  in.minmax <- NULL
  
  # If no min/max target value matrix is given, then get it from train.targets, if provided.
  if ( is.null(minmax.outputs)  && !is.null(t.target)) {
    target.mins <- apply(t.target, 2, min)
    target.maxs <- apply(t.target, 2, max)
    minmax.outputs <- matrix(c(target.mins, target.maxs), ncol=2)
  }
  
  if ( !is.null(train.inputs) ) {
    in.mins <- apply(t.in, 2, min)
    in.maxs <- apply(t.in, 2, max)
    in.minmax <- matrix(c(in.mins, in.maxs), ncol=2)
  }
  
  ciu <- NULL
  in.minmax <- NULL

  # Different models, libraries, ... use different functions for doing "forward pass". 
  model.eval <- function(model, inputs) {
    if ( inherits(model, "FunctionApproximator") ) {
      res <- model$eval(inputs)
    }
    else if ( inherits(model, "train") ) { # caret
      res <- predict(model, inputs, type="prob")
    }
    else {
      # This works at least with "lda" model, don't know with which other ones.
      pred <- predict(model,inputs) 
      res <- pred$posterior
    }
    return(res)
  }
    
  # Calculate Contextual Importance (CI) and Contextual Utility (CU) for the "black-box" bb.
  # "bb" must be an object that supports "eval" method and that only takes a vector/matrix as input
  # Inputs: "black-box" object, input vector, vector of input indices, mx2 matrix of min-max values of outputs
  # "montecarlo.samples" is the number of random values to use for estimating CI and CU.
  # Returns: mx2 matrix with CI, CU for all outputs
  # Might be useful also to return the estimated minimal and maximal values found. Would then be mx2 matrix again
  explain <- function(inputs, ind.inputs.to.explain, in.min.max.limits, montecarlo.samples) {
    last.inputs <<- inputs
    last.explained.inp.inds <<- ind.inputs.to.explain
    if ( !is.null(in.min.max.limits) ) 
      in.minmax <<- in.min.max.limits
    n.mc.samples <<- montecarlo.samples
    
    #contextual.IU<-function(bb, inputs, c.indices, minmax.outputs, montecarlo.samples=100, c.minmax=NULL) {
    # Create matrix of inputs using the provided values, replacing the indicated columns with random values.
    nbr.mc.cols <- length(ind.inputs.to.explain)
    
    # Create random values for the desired columns.
    # Special treatment for [0,1] values, makes it more efficient. 
    if ( is.null(in.minmax) ) {
      rvals <- matrix(runif(n.mc.samples*nbr.mc.cols), nrow=n.mc.samples)
    }
    else {
      # Different treatment required if various min-max ranges for inputs (not [0,1]). 
      mins <- in.minmax[ind.inputs.to.explain,1]
      diffs <- in.minmax[ind.inputs.to.explain,2] - mins
      rvals <- matrix(mins, nrow=n.mc.samples, ncol=nbr.mc.cols, byrow=T) + 
        matrix(runif(n.mc.samples*nbr.mc.cols), nrow=n.mc.samples)*matrix(diffs, nrow=n.mc.samples, ncol=nbr.mc.cols, byrow=T)
    }
    
    # Many/most Machine Learning packages seem to use/require data.frame rather than matrix. 
    if ( is.data.frame(inputs)) { 
      mcm <- inputs[1,] # Initialize as data frame
      mcm[1:n.mc.samples,] <- inputs[1,]
    }
    else { # We try to go with ordinary matrix
      mcm <- matrix(inputs,ncol=length(inputs),nrow=n.mc.samples,byrow=TRUE)
    }
    mcm[,ind.inputs.to.explain] <- rvals
    
    # Evaluate output for all random values, as well as current output. 
    mcout <- model.eval(model, mcm)
    cu.val <- model.eval(model, inputs)
    minvals <- apply(mcout,2,min)
    range <- apply(mcout,2,max) - minvals
    output_ranges <- matrix(minmax.outputs[,2] - minmax.outputs[,1], ncol=1)
    CI <- range/output_ranges
    
    # Calculate CU.
    CU <- (cu.val - minvals)/range
    CU[is.na(CU)] <- 0 # If absmax-absmin was zero
    
    # Finalize the return matrix (maybe data.frame in future?)
    #ciu <- data.frame(CI=CI, CU=CU)
    ciu <- matrix(c(CI,CU), ncol=2)
    colnames(ciu) <- c("CI", "CU")
    if ( !is.null(ciu.row.names) )
      rownames(ciu) <- ciu.row.names
    return(ciu)
  }
  
  # Function for plotting out the effect of changing values of one input on one output
  # bb:
  # etc:
  plot.CI.CU <- function(inputs, ind.input, ind.output, in.min=0, in.max=1, n.points=40, xlab="x", ylab="y", ...) {
    interv <- (in.max - in.min)/n.points
    xp <- seq(in.min,in.max,interv)
    if ( is.null(dim(inputs)) )
      n.col <- length(inputs)
    else
      n.col <- ncol(inputs)
    if ( is.data.frame(inputs)) { 
      m <- inputs[1,] # Initialize as data frame
      m[1:length(xp),] <- inputs[1,]
    }
    else {
      m <- matrix(inputs, ncol=n.col, nrow=length(xp), byrow=T)
    }
    m[,ind.input] <- xp
    yp <- model.eval(model, m)
    cu.val <- model.eval(model, inputs)
    plot(xp, yp[,ind.output], type='l', xlab=xlab, ylab=ylab, ...)
    points(inputs[ind.input], cu.val[ind.output], col = "red", pch = 16, cex = 2)
  }
  
  # Function for 3D plotting the effect of changing values of two inputs on one output
  # bb:
  # ...
  # n.points: How many x/y values for the plot between in.mins and in.maxs.
  # etc:
  plot.CI.CU.3D <- function(inputs, ind.inputs, ind.output, in.mins, in.maxs, n.points=40, ...) {
    interv <- (in.maxs[ind.inputs] - in.mins[ind.inputs])/n.points
    xp <- seq(in.mins[ind.inputs[1]], in.maxs[ind.inputs[1]], by=interv[1])
    yp <- seq(in.mins[ind.inputs[2]], in.maxs[ind.inputs[2]], by=interv[2])
    l <- list(xp,yp)
    pm <- create.permutation.matrix(l)
    if ( is.null(dim(inputs)) )
      n.col <- length(inputs)
    else
      n.col <- ncol(inputs)
    if ( is.data.frame(inputs)) { 
      m <- inputs[1,] # Initialize as data frame
      m[1:nrow(pm),] <- inputs[1,]
    }
    else {
      m <- matrix(inputs, ncol=n.col, nrow=nrow(pm), byrow=T)
    }
    m[,ind.inputs[1]] <- pm[,1]
    m[,ind.inputs[2]] <- pm[,2]
    z <- model.eval(model, m)
    cu.val <- model.eval(model, inputs)
    zm <- matrix(z[,ind.output], nrow = length(xp), byrow = TRUE)
    vt <- persp(xp, yp, zm, ticktype = "detailed", ...) # persp3D might want these: , bg="white", colvar=NULL, col="black", facets=FALSE

    # Show where current instance is located
    x.plot <- as.numeric(inputs[ind.inputs[1]])
    y.plot <- as.numeric(inputs[ind.inputs[2]])
    z.plot <- as.numeric(cu.val[ind.output])
    points(trans3d(x.plot, y.plot, z.plot, pmat = vt), col = "red", pch = 16, cex = 3)
  }
  
  barplot.CI.CU <- function(ci.cu, ind.inputs=NULL, neutral.CU=0.5) {
    if ( is.null(ind.inputs) )
      ind.inputs <- 1:nrow(ci.cu)
    #par(mai=c(0.5,1.2,0,0.1))
    bar.heights <- as.numeric(ci.cu[,1])
    cus <- ci.cu[,2]
    bar.col <- hsv(0.33, 1, cus)
#    barplot(bar.heights,col=c("red","green","yellow","green"),names=c("1","2","3","44444444"),horiz=T,las=1)
    barplot(bar.heights,col=bar.col,horiz=T,las=1)
#    barplot(bar.heights)
  }
   
  # Return list of "public" methods
  pub <- list(
    explain = function(inputs, ind.inputs.to.explain, in.min.max.limits=NULL, montecarlo.samples=100) { 
      explain(inputs, ind.inputs.to.explain, in.min.max.limits, montecarlo.samples)
    },
    plot.CI.CU = function(inputs, ind.input, ind.output, in.min=0, in.max=1, n.points=40, xlab="x", ylab="y", ...) {
      plot.CI.CU (inputs, ind.input, ind.output, in.min, in.max, n.points, xlab, ylab, ...)
    }, 
    plot.CI.CU.3D = function(inputs, ind.inputs, ind.output, in.mins, in.maxs, n.points=40, ...) {
      plot.CI.CU.3D(inputs, ind.inputs, ind.output, in.mins, in.maxs, n.points, ...)
    },
    barplot.CI.CU = function(ci.cu, neutral.CU=0.5) {
      barplot.CI.CU(ci.cu, neutral.CU)
    }
  )
  
  class(pub) <- c("CIU", class(pub))
  return(pub)
}

#=========================================================================
# After this comes development-time code, for testing etc.
#=========================================================================

#source("NeuralLayer.R")
#source("Adaline.R")
#source("RBF.R")

# # Calculate Contextual Importance (CI) and Contextual Utility (CU) for the "black-box" bb.
# # "bb" must be an object that supports "eval" method and that only takes a vector/matrix as input
# # Inputs: "black-box" object, input vector, vector of input indices, mx2 matrix of min-max values of outputs
# # "montecarlo.samples" is the number of random values to use for estimating CI and CU.
# # Returns: mx2 matrix with CI, CU for all outputs
# # Might be useful also to return the estimated minimal and maximal values found. Would then be mx2 matrix again
# contextual.IU<-function(bb, inputs, c.indices, minmax.outputs, montecarlo.samples=100, c.minmax=NULL) {
#   # Create matrix of inputs using the provided values, replacing the indicated columns with random values.
#   n<-montecarlo.samples
#   nbr.mc.cols <- length(c.indices)
#   
#   # Create random values for the desired columns.
#   # Special treatment for [0,1] values, makes it more efficient. 
#   if ( is.null(c.minmax) ) {
#     rvals <- matrix(runif(n*nbr.mc.cols), nrow=n)
#   }
#   else {
#     # Different treatment required if various min-max ranges for inputs (not [0,1]). 
#     mins <- c.minmax[c.indices,1]
#     diffs <- c.minmax[c.indices,2] - mins
#     rvals <- matrix(mins, nrow=n, ncol=nbr.mc.cols, byrow=T) + 
#       matrix(runif(n*nbr.mc.cols), nrow=n)*matrix(diffs, nrow=n, ncol=nbr.mc.cols, byrow=T)
#   }
#   
#   # Many/most Machine Learning packages seem to use/require data.frame rather than matrix. 
#   if ( is.data.frame(inputs)) { 
#     mcm <- inputs[1,] # Initialize as data frame
#     mcm[1:n,] <- inputs[1,]
#   }
#   else { # We try to go with ordinary matrix
#     mcm <- matrix(inputs,ncol=length(inputs),nrow=n,byrow=TRUE)
#   }
#   mcm[,c.indices] <- rvals
#   
#   # Evaluate all output values
#   # Apparently some tweaking required here for different kinds of models. 
#   if ( inherits(bb, "FunctionApproximator") ) {
#     mcout <- bb$eval(mcm)
#     cu.val <- bb$eval(inputs)
#   }
#   else {
#     # This works at least with "lda" model, presumably with "most"(?) R models.
#     pred <- predict(bb,mcm) 
#     mcout <- pred$posterior
#     cv <- predict(bb,inputs)
#     cu.val <- cv$posterior
#   }
#   minvals <- apply(mcout,2,min)
#   range <- apply(mcout,2,max) - minvals
#   output_ranges <- matrix(minmax.outputs[,2] - minmax.outputs[,1], ncol=1)
#   CI <- range/output_ranges
#   
#   # Calculate CU.
#   CU <- (cu.val - minvals)/range
#   result <- matrix(c(CI,CU), ncol=2)
#   #result <- data.frame(CI=CI, CU=CU)
#   colnames(result) <- c("CI", "CU")
#   # Row names are target class names, set them if available. 
#   #if ( !is.null(t.target) && is.data.frame(t.target) && !is.null(colnames(t.target)) )
#   #     rownames(result) <- colnames(t.target)
#   return(result)
# }
# 
# # Function for plotting out the effect of changing values of one input on one output
# # bb:
# # etc:
# plot.CI.CU <- function(bb, instance.values, ind.input, ind.output, in.mins, in.maxs, n.points=40, xlab="x", ylab="y", ...) {
#   interv <- (in.maxs[ind.input] - in.mins[ind.input])/n.points
#   xp <- seq(in.mins[ind.input],in.maxs[ind.input],interv)
#   if ( is.matrix(instance.values) ) 
#     n.col <- ncol(instance.values)
#   else
#     n.col <- length(instance.values)
#   m <- matrix(instance.values, ncol=n.col, nrow=length(xp), byrow=T)
#   m[,ind.input] <- xp
#   yp <- bb$eval(m)
#   plot(xp, yp[,ind.output], type='l', xlab=xlab, ylab=ylab, ...)
#   y <- bb$eval(instance.values)
#   points(instance.values[ind.input], y[ind.output], col = "red", pch = 16, cex = 2)
# }
# 
# # Function for 3D plotting the effect of changing values of two inputs on one output
# # bb:
# # ...
# # n.points: How many x/y values for the plot between in.mins and in.maxs.
# # etc:
# plot.CI.CU.3D <- function(bb, instance.values, ind.inputs, ind.output, in.mins, in.maxs, n.points=40, ...) {
#   interv <- (in.maxs[ind.inputs] - in.mins[ind.inputs])/n.points
#   xp <- seq(in.mins[ind.inputs[1]], in.maxs[ind.inputs[1]], by=interv[1])
#   yp <- seq(in.mins[ind.inputs[2]], in.maxs[ind.inputs[2]], by=interv[2])
#   l <- list(xp,yp)
#   pm <- create.permutation.matrix(l)
#   if ( is.matrix(instance.values) ) 
#     n.col <- ncol(instance.values)
#   else
#     n.col <- length(instance.values)
#   m <- matrix(instance.values, ncol=n.col, nrow=length(xp)*length(yp), byrow=T)
#   m[,ind.inputs[1]] <- pm[,1]
#   m[,ind.inputs[2]] <- pm[,2]
#   z <- bb$eval(m)
#   zm <- matrix(z[,ind.output], nrow = length(xp), byrow = TRUE)
#   vt <- persp(xp, yp, zm, ticktype = "detailed", ...) # persp3D might want these: , bg="white", colvar=NULL, col="black", facets=FALSE
#   x.plot <- instance.values[ind.inputs[1]]
#   y.plot <- instance.values[ind.inputs[2]]
#   z.plot <- bb$eval(matrix(c(x.plot,y.plot), nrow=1))
#   points(trans3d(x.plot, y.plot, z.plot[ind.output], pmat = vt), col = "red", pch = 16, cex = 3)
# }

# Call e.g. "adaline.three.inputs.test()".
# Or "adaline.three.inputs.test(indices=c(1,3))" for getting joint importance of inputs one and three. 
adaline.three.inputs.test <- function(inp=c(0.1,0.2,0.3), indices=c(1), n.samples=100) {
  a <- adaline.new(3, 1)
  inp <- c(0.1,0.2,0.3)
  w <- c(0.20,0.30,0.50)
  a$set.weights(matrix(w, nrow=1, byrow=T))
  ciu <- ciu.new(a, abs.min.max=matrix(c(0, 1), nrow=1, byrow=T))
  CI.CU <- ciu$explain(inp, ind.inputs.to.explain=indices)
  CI.CU
}

## Two outputs
# Call e.g. "adaline.two.outputs.test()"
# Or "adaline.two.outputs.test(indices=c(1,3))" for getting joint importance of inputs one and three. 
adaline.two.outputs.test <- function(inp=c(0.1,0.2,0.3), indices=c(1), n.samples=100) {
  a <- adaline.new(3, 2)
  w <- matrix(c(0.20,0.30,0.50,0.25,0.35,0.40), nrow=2, byrow=TRUE)
  a$set.weights(w)
  #out2 <- a2$eval(inp2)
  ciu <- ciu.new(a, abs.min.max=matrix(c(0,1,0,1), nrow=2, byrow=T))
  CI.CU <- ciu$explain(inp, ind.inputs.to.explain=indices)
  CI.CU
}




