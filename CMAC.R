# "R" implementation of CMAC, done in OOP fashion. 
#
# Kary Främling, created March 2006
#

# Parameters:
# - nbr.layers: number of overlapping "layers"
# - nbr.dimensions: number of input variables
# - minvals: array of minimal values, one per dimension
# - maxvals: array of maximal values, one per dimension
# - nbr.classes: array of number of classes to use, one per dimension
# - random.offset: should layers be offset with random fraction of tile
#   size or evenly. Default TRUE. 
# An extra class or "tile" is added for every dimension so that the layers
# can be offset to the side.
# Only random offset implemented for the moment. 
cmac.new <- function(nbr.layers, nbr.dimensions, minvals, maxvals, nbr.classes, random.offset=TRUE, impose.limits=FALSE) {
  nlayers <- nbr.layers
  ndim <- nbr.dimensions
  mins <- minvals
  maxs <- maxvals
  nclasses <- nbr.classes
  use.random.offset <- random.offset
  outputs <- c()
  inputs <- c()

  # Set up common elements for every layer
  dims <- nbr.classes + 1 # Needed for offseting the layers
  intervals <- (maxvals - minvals)/nbr.classes # Interval sizes

  # Do not use minvals[i] directly - we should offset it with random 
  # or evenly distributed value instead. This is done for each layer
  # separately.
  layers <- list()
  if ( !use.random.offset ) {
    step <- 1/nlayers
    reg.offsets <- seq(0,1-step,step)
  }
  for ( i in 1:nlayers ) {
    if ( use.random.offset ) 
      offsets <- runif(ndim, min=0, max=1)*intervals
    else 
      offsets <-  reg.offsets[i]*intervals
    mi <- mins - offsets
    ma <- maxs - offsets + intervals # We need to add one more class
    layers[[i]] <- discrete.classifier.new(nbr.dimensions, mi, ma, dims, impose.limits)
  }

  # Return vector with one for the correct class in every layer and
  # zero for all other classes. Vectors of all discretizers are combined
  # into one single. 
  get.vector <- function(invals) {
    # Save input values
    inputs <<- invals

    # There must be at least one layer, so this should work
    outputs <<- layers[[1]]$get.vector(inputs) 
    for ( i in 2:nlayers ) {
      outputs <<- c(outputs, layers[[i]]$get.vector(inputs))
    }
    return(outputs)
  }

  # Returns total number of classe, i.e. the length of "outputs"
  # vector. 
  get.total.classes <- function() {
    if ( nlayers == 0 )
      return(0)

    # Same number of classes in all leyers by definition
    return(nlayers*layers[[1]]$get.total.classes()) 
  }
  
  # Construct list of "public methods"
  pub <- list(
              get.nbr.layers = function() { nlayers },
              get.nbr.dimensions = function() { ndim },
              get.minvals = function() { mins },
              get.maxvals = function() { maxs },
              get.nbr.classes = function() { nclasses },
              get.random.offset = function() { use.random.offset },
              get.layers = function() { layers },
              get.vector = function(invals) { get.vector(invals) },
              get.total.classes = function() { get.total.classes() },
              get.inputs = function() { inputs },
              get.outputs = function() { outputs },
              eval = function(invals) { get.vector(invals) }
              )
  
  # We implement "FunctionApproximator"
  fa <- function.approximator.new()
  class(pub) <- c("CMAC",class(fa))
  return(pub)
}

test.CMAC <- function(values, nbr.layers, nbr.dimensions, minvals, maxvals, nbr.classes) {
  source("Functions.R")
  source("DiscreteClassifier.R")
  cmac <- cmac.new(nbr.layers, nbr.dimensions, minvals, maxvals, nbr.classes)
  v <- cmac$get.vector(values)
  print(v)
}

#test.CMAC(c(5,2),5,2,c(-5,-2), c(5,2), c(10,4))
#test.CMAC(c(5,2,3),3,3, c(-5,-2,-3), c(5,2,3), c(10,4,6))

