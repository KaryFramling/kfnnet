# "R" implementation of Adaline, done in OOP fashion. This is
# implemented as a "sub-class" of NeuralLayer, so the only thing to
# add here is training methods. 
#
# Kary Främling, created September 2005
#
adaline.new <- function(nbrInputs, nbrOutputs, use.trace=FALSE) {

  # Set up "super" class and get its environment. We need the environment
  # only if we need to access some variables directly, not only through
  # the "public" interface.
  # "super" has to have at least one "method" for getting the environment.
  super <- neural.layer.new(nbrInputs, nbrOutputs, weighted.sum.activation, identity.output.function, use.trace)
  se <- environment(super[[1]])

  # Set up our "own" instance variables.
  nlms <- F # Use standard LMS by default
  mixtw <- 1.0 # Weight of this Adaline when used in some
               #kind of "mixture model", e.g. BIMM

  # Perform LMS or NLMS training. No return value.
  # CAN THIS BE APPLIED TO ONLY ONE SAMPLE OR A WHOLE SET???
  train <- function(t) {
    targets <<- t
    # Widrow-Hoff here
    if ( !nlms ) {
      delta <- as.vector(lr*(targets - outputs))%o%as.vector(inputs)
    }
    else {
      nfact <- matrix(mixtw*(inputs%*%t(inputs)),
                      nrow=nrow(inputs), ncol=ncol(inputs))
      delta <- as.vector(lr*(targets - outputs))%o%as.vector(inputs/nfact)
    }
    weights <<- weights + delta
  }

  # Perform LMS or NLMS training for given "delta" value that indicates
  # the error. "delta" can be a constant.
  # Should be modified so that it can also be a vector. Having a vector
  # is useful for "ordinary" learning where the error value is usually
  # different for every output. 
  # In Q-learning, the error value is global for the whole net, i.e. for
  # all outputs. Then the extent to which this error is distributed to
  # different outputs depends on the "eligibility trace". The 
  # the expression for "delta" is:
  # r(t+1) + gamma*Q(s(t+1),a(t+1)) - Q(s(t),a(t))
  # IMPORTANT!?? This function can only be used for discrete
  # state spaces with lookup-table type of calculations. This means
  # that sum(inputs^2) = 1, so NLMS only signifies using "K" factor. 
  train.with.delta <- function(delta) {
    if ( is.null(trace) ) {
      tr <- 1.0
    }
    else {
      tr <- trace$get.trace()
    }
    d <- lr*delta*tr
    if ( nlms ) {
      nfact <- matrix(mixtw*(inputs%*%t(inputs)), nrow=nrow(d), ncol=ncol(d))
      d <- d/nfact
    }
    
    weights <<- weights + d
  }

  # Construct list of "public methods"
  pub <- list(
              get.nlms = function() { nlms },
              get.mixtw = function() { mixtw },
              set.nlms = function(value) { nlms <<- value },
              set.mixtw = function(value) { mixtw <<- value },
              train.with.delta = function(delta) {
                train.with.delta(delta)
              },
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
  class(methods) <- c("Adaline",class(super),class(trainable.approximator.new()))

  return(methods)
}

test <- function() {
  a <- adaline.new(2, 3)
  class(a)
  print(class(a))
  print(a)
  print(a$get.inputs())
  inputs <- c(1, 1.5)
  weights <- matrix(c(0, 0, 1, 1, 2, 2), 3, 2, byrow = T)
  out <- a$eval(inputs)
  print(out)
  a$set.weights(weights)
  out <- a$eval(inputs)
  print(out)
  print(a$get.weights())
  print(a$get.inputs())
  print(a$get.outputs())
  print(a$get.lr())
  print(a$set.lr(0.5))
  print(a$get.lr())
  print(a$get.inputs())
  t <- c(1, 2, 3)
  a$set.nlms(T)
  a$train(t)
  out <- a$eval(inputs)
  print(out) # Should give 0.5, 2.25, 4
  print(a$get.targets()) # Should give i, 2, 3
}

#test()
