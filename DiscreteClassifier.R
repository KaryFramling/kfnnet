# Discrete multi-dimensional classifier, done in OOP fashion. 
#
# Kary Främling, created 16 March 2006
#

# Create new "discrete classifier object"
# Parameters:
# - nbr.dimensions: number of input variables
# - minvals: array of minimal values, one per dimension
# - maxvals: array of maximal values, one per dimension
# - nbr.classes: array of number of classes to use, one per dimension
discrete.classifier.new <- function(nbr.dimensions, minvals, maxvals, nbr.classes, impose.limits=FALSE) {
  ndim <- nbr.dimensions
  mins <- minvals
  maxs <- maxvals
  nclasses <- nbr.classes
  total.classes <- 1
  inputs <- c()
  for ( i in 1:nbr.dimensions )
    total.classes <- total.classes*nclasses[i]

  outputs <<- vector(mode = "integer", length = total.classes)

  # "impose.limits" can be one value or a vector of values
  # Here we make it a vector of right length and values
  imp.limits <- vector(mode="logical", length=ndim)
  if ( length(impose.limits) > 1 ) # Is it a vector already?
    imp.limits[1:length(impose.limits)] <- impose.limits[1:length(impose.limits)]
  else
    imp.limits[] <- impose.limits

  discretizers <- list()
  for ( i in 1:nbr.dimensions ) {
    discretizers[[i]] <- discretizer.new(minvals[i], maxvals[i], nbr.classes[i], imp.limits[i])
  }

  # Get class index as a function of the values in the "inputs" parameter,
  # whose length should correspond to the number of dimensions.
  # Zero is returned if any of the values does not fit into one of the
  # classes
  get.index <- function(invals) {
    inputs <<- invals
    ind <- discretizers[[1]]$get.class(inputs[1])
    if ( ind == 0 ) # Some value did not fit anywhere
      return(0)
    mult <- 1
    for ( i in 2:nbr.dimensions ) {
      mult <- mult*nclasses[i-1]
      dind <- discretizers[[i]]$get.class(inputs[i])
      if ( dind == 0 ) # Some value did not fit anywhere
        return(0)
      ind <- ind + (dind-1)*mult
    }
    return(ind)
  }

  # Return vector that is one for the correct class and zero for all
  # other classes. 
  get.vector <- function(invals) {
    inputs <<- invals
    outputs[] <<- 0
    ind <- get.index(inputs)
    if ( ind > 0 ) # Did any class fit?
      outputs[get.index(inputs)] <<- 1
    return(outputs)
  }
  
  set.discretizer <- function(index, discretizer) {
    discretizers[[index]] <<- discretizer
    nclasses[index] <<- discretizer$get.nbr.classes()
    total.classes <<- 1
    for ( i in 1:nbr.dimensions )
      total.classes <<- total.classes*nclasses[i]
  }

  # Construct list of "public methods"
  pub <- list(
              get.nbr.dimensions = function() { ndim },
              get.minvals = function() { mins },
              get.maxvals = function() { maxs },
              get.nbr.classes = function() { nclasses },
              get.total.classes = function() { total.classes },
              get.index = function(invals) { get.index(invals) },
              get.vector = function(invals) { get.vector(invals) },
              eval = function(invals) { get.vector(invals) },
              get.inputs = function() { inputs },
              get.outputs = function() { outputs },
              get.discretizer = function(index) { discretizers[[index]] },
              set.discretizer = function(index, discretizer) {
                set.discretizer(index, discretizer)
              }
              )
  
  # We implement "FunctionApproximator"
  fa <- function.approximator.new()
  class(pub) <- c("DiscreteClassifier",class(fa),class(pub))
  return(pub)
}

test.discrete.classifier <- function(values, nbr.dimensions, minvals, maxvals, nbr.classes) {
  source("Functions.R")
  dc <- discrete.classifier.new(nbr.dimensions, minvals, maxvals, nbr.classes)
  print(dc$get.index(values))
  print(dc$get.vector(values))
}

#test.discrete.classifier(c(5,2),2,c(-5,-2), c(5,2), c(10,4))
#test.discrete.classifier(c(-5,2,3),3, c(-5,-2,-3), c(5,2,3), c(10,4,6))

