# Lookup-table implementation of action-value function estimation.
# This is here implemented by a task-dependent discretisation object
# that transforms (if needed) the actual state variable values into
# a state index.
# The lookup-table is implemented as an Adaline even though it is not
# the most efficient way of doing it. But it's practical for the moment
# because it makes the implementation very similar to function approximators. 
lookup.table.estimator.new <- function(nbrInputs, nbrOutputs, discretizer) {
  nin <- nbrInputs
  nout <- nbrOutputs
  inputs <- c()
  outputs <- c()
  targets <- c()
  discrete.inputs <- vector("integer", length=nin)
  state <- -1
  discr <- discretizer
  lookup.table <- adaline.new(nin, nout)

  # Takes current values in "inputs" and sets "discrete.inputs" to
  # corresponding values. If the discretizer is null, then just copy
  # the values directly
  set.discrete.inputs <- function() {
    if ( is.null(discr) ){
      discrete.inputs <<- inputs
    }
    else {
      state <<- discr$get.class(inputs)
      discrete.inputs[] <<- 0
      discrete.inputs[state] <<- 1
    }
  }

  # Evaluate input values and return vector of output values.
  eval <- function(invals) {
    inputs <<- invals
    set.discrete.inputs()
    outputs <<- lookup.table$eval(discrete.inputs)
    return(outputs)
  }

  train <- function(t) {
    targets << t
    lookup.table$train(targets)
  }
  
  # Return list of "public" methods
  list(
       get.inputs = function() { inputs },
       get.outputs = function() { outputs },
       get.state = function() { state },
       get.lookup.table = function() { lookup.table },
       set.lookup.table = function(o) { lookup.table <<- o },
       get.discretizer = function() { discr },
       set.discretizer = function(o) { discr <<- o },
       eval = function(invals) { eval(invals) }
       train = function(t) { train(t) }
       )  
}
