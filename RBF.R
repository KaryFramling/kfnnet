# "R" implementation of RBF Neural Network, done in OOP fashion.
#
# Kary Fr?mling, created 28 dec 2005
#

# Everything needed for this file to work
source("Functions.R")
source("Interfaces.R")
source("NeuralLayer.R")
source("Adaline.R")
source("RBFclassifier.R")

# Create new instance of RBF network, with following properties: 
# nbrInputs: Number of inputs
# nbrOutputs: Number of outputs
# n.hidden.neurons: Number of hidden neurons (allowed to provide zero as value)
# activation.function: The activation function to be used by hidden layer neurons
# output.function: Output function to use for hidden layer neurons
# normalize: Normalize hidden layer outputs or not, i.e. "ordinary" RBF or normalized RBF
rbf.new <- function(nbrInputs, nbrOutputs, n.hidden.neurons,
                    activation.function=squared.distance.activation,
                    output.function=gaussian.output.function,
                    normalize=FALSE) {

  ninputs <- nbrInputs
  noutputs <- nbrOutputs
  nhidden <- n.hidden.neurons
  outputs <- c()
  inputs <- c()
  targets <- c()

  # Create neural layers
  outlayer <- adaline.new(nhidden, noutputs)
  hidden <- rbf.classifier.new(ninputs, nhidden, activation.function, output.function)
  hidden$set.normalize(normalize)

  # Set default parameters
  hidden$set.spread(1.0)
  
  # Method for doing forward-pass, i.e. calculate outputs based on inputs
  eval <- function(invals) {
    inputs <<- invals
    h <- hidden$eval(inputs)
    outputs <<- outlayer$eval(h)
    return(outputs)
  }

  # There are many possible ways of training. This one is for
  # Widrow-Hoff on output layer. An "eval" must
  # always be performed prior to calling "train" in order to set input
  # values and evaluated values correctly.
  # The number of rows in "input" matrix has to be the same as in the
  # "target" matrix.
  train <- function(t) {
    targets <<- t
    outlayer$train(t)
  }
  
  # Return list of "public" methods
  pub <- list(
              get.inputs = function() { inputs },
              get.outputs = function() { outputs },
              get.outlayer = function() { outlayer },
              get.hidden = function() { hidden },
              get.spread = function() { hidden$get.spread() },
              get.nrbf = function() { hidden$get.normalize() },
              set.outlayer = function(o) { outlayer <<- o },
              set.hidden = function(o) { hidden <<- o },
              set.spread = function(value) { hidden$set.spread(value) },
              set.nrbf = function(value) { hidden$set.normalize(value) },
              eval = function(invals) { eval(invals) },
              train = function(t) { train(t) }
  )

  # We implement "FunctionApproximator"
  fa <- function.approximator.new()

  class(pub) <- c("RBF",class(fa),class(pub))
  return(pub)
  
}

# INKA training algorithm, as in my (Kary's) PhD thesis. Parameters:
# - rbf: RBF network instance to use
# - train.inputs: Matrix of input values of the training set
# - train.outputs: Matrix of target output values of the training set
# - c: "c" paremeter in INKA, i.e. minimal allowed distance between neurons
# - max.iter: Maximal number of training iterations
# - inv.whole.set.at.end: If TRUE, then pseudo-matrix is calculated for entire traning set at the end. 
#   During training it is done only for the examples used to create hidden neurons.
# - classification.error.limit: If specified, training ends when the number of classification errors 
#   goes under this value. To be used for classification tasks. 
# - rmse.limit: If specified, traning ends when RMSE between traget values and output values goes under (or equals)
#   the given limit. 
train.inka <- function(rbf, train.inputs, train.outputs, c=0, max.iter=1, 
                       inv.whole.set.at.end=T, classification.error.limit=NULL, rmse.limit=NULL) {
  
  # Initialize set of remaining (not yet used for creating hidden neuron) traning examples. 
  # Set of used examples is initially empty.
  t.in.remain <- as.matrix(train.inputs) # The set of remaining/unused training examples. 
  targets.remain <- as.matrix(train.outputs) # as.matrix is to deal with case of it being a vector
  t.in.used <- NULL
  targets.used <- NULL
  
  # Loop until stopping condition, e.g. on some error measure, number of hodden neurons or whatever. 
  for ( iter.ctr in 1:max.iter ) {
    ty <- rbf$eval(t.in.remain) # Do forward pass for all "remaining" training examples
    
    # Find out which one has the greatest error, taken over all outputs
    diff <- abs(targets.remain - ty) # Estimation error for whole training set, absolute value
    rdiff <- apply(diff, 1, max) # We only want to know which rows have greatest error
    max.error <- max(rdiff) # Largest error
    indices <- which(rdiff %in% max.error) # Indices of rows with largest error
    n.max <- length(indices)
    if ( n.max > 1 ) { # If more than one row with largest error, then take a random one of them. 
      i <- 1 + round(runif(1)*n.max)
      if ( i > n.max ) 
        i <- n.max # Just for unprobable case of runif = 1
      ind.larg.err <- indices[i]
    } else {
      ind.larg.err <- indices[1]
    }
    
    # Add new hidden neuron
    new.centroid <- t.in.remain[ind.larg.err,]
    new.tout <- targets.remain[ind.larg.err,]
    
    # Create set of "used" examples if this is the first one, otherwise append new example to it. 
    if ( is.null(t.in.used) ) { 
      t.in.used <- matrix(new.centroid, nrow=1)
      targets.used <- matrix(new.tout, nrow=1)
    } else { 
      t.in.used <- rbind(t.in.used, new.centroid)
      targets.used <- rbind(targets.used, new.tout)
    } 
    
    # Remove from set of remaining/unused training examples. 
    t.in.remain <- as.matrix(t.in.remain[-ind.larg.err,])
    targets.remain <- as.matrix(targets.remain[-ind.larg.err,])
    
    # Add new hidden neuron and output weight. But only if far enough from existing centroids
    # TO BE IMPLEMENTED!
    # if ( new.centroid far enough from existing ones ) {}
    # Get distance of hidden neuron centroids to all data points. 
    #    hw <- hl$get.weights()
    #    ref <- hw[1,]
    #    t.aff <- t(aff.trans$eval(t(t.in)))
    #    apply(t.aff,1,function(x)sqrt(sum((x-ref)^2)))
    
    # Add one more column to outpur layer weights, initialize to zero. 
    ol <- rbf$get.outlayer()
    ow <- ol$get.weights()
    ow <- cbind(ow, 0)
    ol$set.weights(ow)
    
    # Add one more neuron (row) to hidden layer with centroid initialized to training example.
    hl <- rbf$get.hidden()
    hw <- hl$get.weights()
    hw <- rbind(hw, new.centroid)
    hl$set.weights(hw)
    
    # Do forward pass of RBF net for re-calculating hidden layer outputs. 
    ty <- rbf$eval(t.in.used)
    h.out <- hl$get.outputs()
    
    # Get pseudo-inverse matrix between hidden layer outputs and the correct/target values 
    # and use that as new weights for output layer. 
    w <- qr.solve(h.out, targets.used)
    ol$set.weights(t(w))
    
    # Check if classification error goal has been achieved, for the moment 
    # for training set only.  
    if ( !is.null(classification.error.limit) ) {
      y <- rbf$eval(train.inputs)
      # If there's only one output, then there are two classes indicated by 0 or 1 value. 
      # Then classification error is calculated differently
      if ( ncol(train.outputs) == 1) {
        nbr.errors <- sum(abs(train.outputs - round(y)))
      }
      else {
        classification <- (y == apply(y, 1, max)) * 1
        nbr.errors <- sum(abs(train.outputs - classification)) / 2 # One error gives sum on 2. 
      }
      # Stop training if error is small enough.
      if ( nbr.errors <= classification.error.limit ) 
        break
    }
    
    # Check if RMSE goal has been achieved, on training set for the moment.
    if ( !is.null(rmse.limit) ) {
      y <- rbf$eval(train.inputs)
      rmse <- root.mean.squared.error(train.outputs, y)
      if ( rmse <= rmse.limit ) {
        break
      }
    }
  }
  
  # Do matrix inversion for whole training set at the end
  if ( inv.whole.set.at.end ) {
    ty <- rbf$eval(train.inputs)
    h.out <- hl$get.outputs()
    w <- qr.solve(h.out, train.outputs)
    ol$set.weights(t(w))
  }
  return(nrow(hl$get.weights())) # Return number of hidden neurons created
}


# Create an RBF network "manually", i.e. populate network and initialize weights "by hand". 
test.rbf <- function() {
  #rbf <- rbf.new(1, 1, 0, activation.function=squared.distance.activation, 
  #               output.function=gaussian.output.function)
  rbf <- rbf.new(1, 1, 0, activation.function=squared.distance.activation, 
                 output.function=imqe.output.function)
  ol <- rbf$get.outlayer()
  ol$set.weights(matrix(c(1,2),nrow=1,ncol=2))
  hl <- rbf$get.hidden()
  hl$set.weights(matrix(c(1,5),nrow=2,ncol=1,byrow=T))
  rbf$set.nrbf(TRUE)
  rbf$set.spread(1)
  x <- matrix(seq(-5,10,0.1))
  y <- rbf$eval(x)
  #for ( i in x )
  #  print(rbf$eval(i))
  plot(x, y, type='l')
}

#source("Functions.R")
#source("Adaline.R")
#source("NeuralLayer.R")
#test.rbf()

