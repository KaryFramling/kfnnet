# "R" implementation of RBF "hidden layer", done in OOP fashion.
#
# Kary Främling, created 5 sep 2006
#

rbf.classifier.new <- function(nbrInputs, nbrOutputs, 
                               activation.function=squared.distance.activation,
                               output.function=gaussian.output.function) {

  # Set up "super" class and get its environment. We need the environment
  # only if we need to access some variables directly, not only through
  # the "public" interface.
  # "super" has to have at least one "method" for getting the environment.
  super <- neural.layer.new(nbrInputs, nbrOutputs, activation.function, output.function)
  se <- environment(super[[1]])

  # Then set up our "own" instance variables. 
  norm.inps <- FALSE
  aff.trans <- NULL

  # Vector of input indices where input values are "wrapped" from
  # smallest to greatest as for the angle in pendulum task.
  # NOT IMPLEMENTED YET as vector, for the moment this can be an
  # input index or NULL if not applicable. 
  wrapped.inputs <- NULL 
  mins <- NULL
  maxs <- NULL

  # Evaluate "invals" and return corresponding "outvals". If no neural
  # layer has been created yet, then just return zero. 
  eval <- function(invals) {
    nsamples <- nrow(invals)
    if ( is.null(nsamples) )
      nsamples <- 1
    out <- matrix(0, nrow=nsamples, ncol=nrow(weights))
    for ( r in 1:nsamples ) { #Have to do as a loop, would be better to change into single evaluations!
      if ( nsamples > 1 )
        ins <- invals[r,]
      else
        ins <- invals
      if ( is.null(aff.trans) ) # Apply affine transformation if any
        inputs <<- ins
      else
        inputs <<- t(aff.trans$eval(ins))
      if ( !is.null(wrapped.inputs) ) 
        out[r,] <- eval.with.wrap(inputs)
      else
        out[r,] <- super$eval(inputs)
    }
    outputs <<- out # "outputs" was modified in "super$eval"!
    return(outputs)
  }

  # With wrapped inputs, we evaluate one more range down and up so that
  # we are sure to activate all neurons that need to be activated.
  # IMPORTANT! For the moment, only one wrapped input is supported,
  # i.e. the one whose index is the first element in wrapped.inputs.
  # This might be useful to improve in future tasks, let's see. 
  eval.with.wrap <- function(invals) {
    wi <- wrapped.inputs[1]
    range <- maxs[wi]-mins[wi]
    positive <- negative <- invals;
    negative[wi] <- invals[wi]-range
    positive[wi] <- invals[wi]+range

    # Remove normalisation at this phase (if activated), then evaluate
    is.norm <- normalize
    normalize <<- FALSE
    out <- super$eval(invals)
    out <- pmax(out, super$eval(negative))
    out <- pmax(out, super$eval(positive))

    # Then perform necessary extra operations in case of NRBF
    if ( is.norm ) {
      out <- normalize.to.one(out)      
      normalize <<- TRUE
    }
    outputs <<- out
    return(outputs)
  }
  
  # Initialize centroids so that they form a uniform grid. The three
  # parameters are:
  # - minvals: array of minimal values expected for each input.
  # - maxvals: array of maximal values expected for each input.
  # - nbr.classes: how many centroid values will be used for each
  #   input dimension. The interval from min to max is divided into
  #   that many classes and the centroid values become the median
  #   values for each class.
  # The "nbrOutputs" property obviously has to fit with the number
  # of classes multiplied together. In practice, the corresponding
  # neural layer is re-created in order to be sure to satisfy this
  # condition. 
  init.centroids.grid <- function(minvals, maxvals, nbr.classes,
                                  affine.transformation=NULL) {

    # Store affine transformation object if any
    aff.trans <<- affine.transformation
    mins <<- minvals
    maxs <<- maxvals
    if ( !is.null(aff.trans) ) {
      mins <<- aff.trans$eval(mins)
      maxs <<- aff.trans$eval(maxs)
    }

    # Calculate number of neurons needed
    noutputs <- 1
    for ( i in 1:length(nbr.classes) )
      noutputs <- noutputs*nbr.classes[i]

    # Create list of vectors of centroid values to use
    centroid.values <- list()
    for ( i in 1:length(minvals) ) {
      cv <- vector(mode="numeric", length=nbr.classes[i])
      interval <- (maxs[i] - mins[i])/nbr.classes[i]
      cval <-  mins[i] + interval/2
      for ( ci in 1:nbr.classes[i] ) {
        cv[ci] <- cval
        cval <- cval + interval
      }
      centroid.values[[i]] <- cv
    }

    # Create corresponding weight matrix - one row per output, one column
    # per input. There's probably some neat functions for doing this directly
    # but didn't find them so had to create own one...
    w <- create.permutation.matrix(centroid.values)
    weights <<- w
  }

  train <- function(t) {
    
  }
  
  # Return list of "public" methods
  pub <- list(
              get.normalize.inputs = function() { normalize.inputs },
       #train.with.trace = function(diff, trace) {
       #  train.with.trace(diff, trace)
       #},
              init.centroids.grid = function(minvals, maxvals, nbr.classes,
                affine.transformation=NULL) {
                init.centroids.grid(minvals, maxvals, nbr.classes,
                                    affine.transformation)
              },
              get.affine.transformation = function() { affine.transformation },
              set.affine.transformation = function(at) {
                affine.transformation <<- at
              },
              get.wrapped.inputs = function() { wrapped.inputs },
              set.wrapped.inputs = function(v) { wrapped.inputs <<- v },
              get.total.classes = function() { nrow(weights) },
              get.vector = function(invals) { eval(invals) },
              eval = function(invals) { eval(invals) }, # Override this one.
              train = function(t) { train(t) }
              )


  # Set up the environment so that "private" variables in "super" become
  # visible. This might not always be a good choice but it is the most
  # convenient here for the moment. 
  parent.env(environment(pub[[1]])) <- se

  # We return the list of "public" methods. Since we "inherit" from
  # "super" (NeuralLayer), we need to concatenate our list of "methods"
  # with the inherited ones.
  # Also add something to our "class". Might be useful in the future
  methods <- c(pub,super)

  # Also declare that we implement the "TrainableApproximator" interface
  class(methods) <- c("RBFclassifier",class(super),class(pub))
  return(methods)
  
}

test.rbf.classifier <- function() {
  rbfc <- rbf.classifier.new(nbrInputs=2, nbrOutputs=0, 
                             activation.function=squared.distance.activation,
                             output.function=gaussian.output.function,
                             normalize=FALSE)
  rbfc$init.centroids.grid(c(-1,-1), c(1,1), c(2,2))
  rbfc$set.spread(1)
  rbfc$set.normalize(TRUE)
  x <- seq(-1.5,1.5,0.1)
  y <- x
  inps <- create.permutation.matrix(list(x,y))
  z <- rbfc$eval(inps)
  par(mfcol=c(2,2))
  for ( i in 1:4 ) {
    zplot <- matrix(z[,i], nrow=length(x), ncol=length(y))
    persp(x,y,zplot,theta=135,phi=45)
  }
  x11()
  zplot <- matrix(rowSums(t(t(z)*c(-1,1,2,4))), nrow=length(x), ncol=length(y))
  persp(x,y,zplot,theta=135,phi=45)
  #for ( i in x )
    #print(rbfc$eval(i))
  #plot(x, y, type='l')
}


#test.rbf.classifier()

