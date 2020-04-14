# "R" implementation of Neural Network layer, done in OOP fashion.
#
# Kary Fr?mling, created 28 dec 2005
#

#-----
# "Activation" functions here, i.e. the one that determines activation level
# as a function of input values.
#-----

# Weighted sum, the most common activation function.
weighted.sum.activation <- function(input, weights) {
  res <- crossprod(t(input),t(weights))
  res  
}

# Squared Euclidean distance between weights and inputs.
squared.distance.activation <- function(input, weights) {
  # Loops are bad but I can't find better solutions just now...
  d2 <- matrix(0,nrow=nrow(input),ncol=nrow(weights))
  for ( i in 1:nrow(input) ) {
    diff <- weights - matrix(input[i,], nrow=nrow(weights), ncol=ncol(weights), byrow=T)
    d2[i,] <- rowSums(diff*diff)
  }
  d2
}

#-----
# "Output" functions here, i.e. functions that take activation value
# and transform it into output value.
#-----

# Output is same as activation. 
identity.output.function <- function(activation) {
  activation
}

# Gaussian function. Spread parameter may
# be a single value or a vector, i.e. same spread for all or
# individual spread parameters. The activation value should normally
# be the squared distance. 
sigmoid.output.function <- function(activation) {
  res <- 1/(1+exp(-activation))
  res
}

# Gaussian function. Spread parameter may
# be a single value or a vector, i.e. same spread for all or
# individual spread parameters. The activation value should normally
# be the squared distance. 
gaussian.output.function <- function(activation, spread=1.0) {
  res <- exp(-activation/spread)
  res
}

# Inverse MultiQuadric Equations function. Spread parameter may
# be a single value or a vector, i.e. same spread for all or
# individual spread parameters. The activation value should normally
# be the squared distance. 
imqe.output.function <- function(activation, spread=1.0) {
  res <- 1/sqrt(1 + activation/spread)
  res
}

#-----
# Then we also need "inverse" functions for gradient descent.
#-----

# Neural layer object implementation. 
neural.layer.new <- function(nbrInputs, nbrOutputs, activation.function, output.function, use.trace=FALSE, use.bias=FALSE) {

  weights <- matrix(0, nrow = nbrOutputs, ncol = nbrInputs)
  afunction <- activation.function
  ofunction <- output.function
  activations <- c()
  outputs <- c()
  inputs <- c()
  targets <- c()
  lr <- 0.1 # Learning rate
  spread <- 1.0 # For Kernel output functions
  normalize <- FALSE # No normalisation of output values by default
  
  # "Eligibility trace" for reinforcement learning purposes.
  if ( use.trace )
    trace <- eligibility.trace.new(nbrInputs, nbrOutputs)
  else
    trace <- NULL

  # Evaluate output values for the given input values. The input
  # values are organised in columns, one row per sample. 
  eval <- function(invals) {
    if ( is.matrix(invals) )
      inputs <<- invals
    else
      inputs <<- matrix(invals, nrow=1)

    if ( use.bias ) { # Add one more column to inputs. Current solution seems clumsy but will have to do for the moment...
      dinp <- dim(inputs)
      bias.matrix <- matrix(1, dinp[1], dinp[2]+1)
      bias.matrix[1:dinp[1],1:dinp[2]] <- inputs
      inputs <<- bias.matrix
    }
    activations <<- afunction(inputs, weights)
    if ( sum(names(formals(ofunction)) == "spread") > 0 )
      outputs <<- ofunction(activations, spread)
    else
      outputs <<- ofunction(activations)
    if ( normalize ) 
       outputs <<- normalize.to.one(outputs)
    return(outputs)
  }
  
  # Return list of "public" methods. Also set "class" attribute.
  pub <- list(
              eval = function(invals) { eval(invals) },
              get.inputs = function() { inputs },
              get.outputs = function() { outputs },
              get.targets = function() { targets },
              get.weights = function() { weights },
              get.trace = function() { trace },
              get.lr = function() { lr },
              get.spread = function() { spread },
              get.normalize = function() { normalize },
              get.use.bias = function() { use.bias },
              set.weights = function(w) { weights <<- w },
              set.trace = function(tr) { trace <<- tr },
              set.lr = function(lrate) { lr <<- lrate },
              set.spread = function(s) { spread <<- s },
              set.normalize = function(value) { normalize <<- value },
              set.use.bias = function(value) { use.bias <<- value }
              )

  # We also implement "FunctionApproximator"
  fa <- function.approximator.new()

  class(pub) <- c("NeuralLayer",class(fa))
  return(pub)
}

test <- function() {
  l <- neural.layer.new(2, 1, weighted.sum.activation, identity.output.function)
  l$set.weights(matrix(c(1, 2), nrow=1, ncol=2))
  out <- l$eval(c(1, 1))
  print(out)

  # Test squared distance
  l <- neural.layer.new(1, 2, squared.distance.activation, imqe.output.function)
  l$set.weights(matrix(c(1, 2), nrow=2, ncol=1, byrow=T))
  l$set.spread(1.0)
  out <- l$eval(c(2))
  print(out)

}

#test()
#s<-seq(0,100,0.1)
#plot(s,imqe.output.function(s),type='l', col='black', ylim=c(0,1))
#lines(s,gaussian.output.function(s),col='green')
#s<-seq(-5,5,0.1)
#plot(s,sigmoid.output.function(s),type='l', col='black', ylim=c(0,1))
#d <- matrix(c(1,2,2,3),ncol=2,byrow=T)
#d <- matrix(c(1,2),ncol=2,byrow=T)
#w <- matrix(c(1,1,2,2,3,3),ncol=2,byrow=T)
#a <- weighted.sum.activation(d,w)
#a <- squared.distance.activation(d,w)
#print(a)
#o <- gaussian.output.function(a, spread=1.0)
#print(o)

