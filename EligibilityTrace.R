# Eligibility trace for the weights of one neural layer, i.e. in
# principle one weight matrix. Can also be used for lookup-table
# representations. 
#
eligibility.trace.new <- function(ninputs, noutputs, lambda.decay=0.0) {
  nin <- ninputs
  nout <- noutputs
  trace <- matrix(0, nrow=nout, ncol=nin)
  lambda <- lambda.decay # Trace decay rate
  discount.rate <- 1.0 # Discount rate for discounted tasks. 1.0: no discount
  invector <- vector(mode='numeric', length=nin)
  outvector <- vector(mode='numeric', length=nout)
  reset.unused.actions <- FALSE
  accumulating.trace <- FALSE # Replacing trace is default
  
  decay <- function() {
    trace <<- discount.rate*lambda*trace
  }

  # Set current state and action for discrete (lookup-table) representations.
  set.current.lookup.table <- function(state, action) {
    decay()
    if ( reset.unused.actions ) {
      trace[, state] <<- 0.0 # as in Singh&Sutton, 1996
    }
    trace[action, state] <<- 1.0
  }
  
  # Set current (discrete) action for current (possibly continuous) state.
  # "action" is therefore an index, while "inputs" is a vector.
  set.current.action <- function(inputs, action) {
    outvector[] <<- 0.0
    outvector[action] <<- 1.0
    set.current(inputs, outvector)
  }
  
  # Update trace for given input vector and the output vector
  # that corresponds to what extent every output corresponds to
  # the taken action. In the discrete-action case, one output is
  # one and the others zero. 
  set.current <- function(inputs, outputs) {
    invector <<- inputs
    outvector <<- outputs
    decay()

    # A "compromise" solution that should work for discrete case, CMAC,
    # continuous-valued etc. as in Singh&Sutton, 1996. We choose to
    # set trace to zero for all actions (outputs) with values < 1 and
    # all state variables with values > 0.
    if ( !accumulating.trace && reset.unused.actions ) {
      ainds <- outputs >0
      trace[outputs<1,inputs>0] <<- 0.0
    }

    # Here we need to make changes - there is not just one input
    # that can be active. Both additive and replacing eligibility
    # traces can be used. The biggest challenge is how to dispatch
    # eligibility to different weights.
    # 1. We assume that we have a total eligibility to distribute
    #    equal to one. For a lookup-table, this will then be identical
    #    to the classical eligibility trace (but do remember to use
    #    dr*lambda as lambda value here for such use!)
    # 2. Eligibility value is "relative contribution of this output
    #    for the actual output"*input activation value.
    # Now, here is an inconsistency: there should be a division of
    # the trace values here by their sum so that the sum of the trace
    # update would be one. But this will require some further studies,
    # it sounds quite OK if we only have continuous features (but
    # binary actions) and as long as the ANN is "well-behaving" in
    # some way. But what will happen e.g. with badly positioned RBFs
    # without normalisation? 
    dim(inputs) <- NULL
    t <- outer(outputs,inputs)

    # 3. Accumulatng eligibility trace: add resulting value to
    #    previous value
    #    Replacing eligibility trace: take max-value between
    #    new value and old value.
    # We use "replacing" version as default, i.e. max-operator.
    if ( accumulating.trace )
      trace <<- trace + t
    else
      trace <<- pmax(trace,t) 
  }
  
  # Return list of "public methods"
  list (
        reset = function() { trace[,] <<- 0 },
        get.nin = function() { nin },
        get.nout = function() { nout },
        get.trace = function() { trace },
        get.lambda = function() { lambda },
        set.lambda = function(value) { lambda <<- value },
        get.dr = function() { discount.rate },
        set.dr = function(value) { discount.rate <<- value },
        get.reset.unused.actions = function() { reset.unused.actions },
        set.reset.unused.actions = function(value) { reset.unused.actions <<- value },
        get.accumulating.trace = function() { accumulating.trace },
        set.accumulating.trace = function(value) { accumulating.trace <<- value },
        get.value = function (input, output) {
          trace[output, input]
        },
        set.current.lookup.table = function(state, action) {
          set.current.lookup.table(state, action)
        },
        set.current.action = function(input, action) {
          set.current.action(input, action)
        },
        set.current = function(input, output) { set.current(input, output) },
        decay = function() { decay() }
        )
}

test.discrete.trace <- function() {
  wet <- eligibility.trace.new(3, 3, lambda.decay=0.5)
  wet$set.reset.unused.actions(TRUE)
  wet$set.current.action(c(0,1,0),2)
  print(wet$get.trace())
  wet$decay()
  print(wet$get.trace())
  wet$set.current(c(0,1,0),c(0,0,1))
  print(wet$get.trace())
  wet$decay()
  print(wet$get.trace())
  wet$set.current.action(c(1,0,0),3)
  print(wet$get.trace())
  wet$decay()
  print(wet$get.trace())
  wet <- eligibility.trace.new(3, 3, lambda.decay=0.5)
  wet$set.reset.unused.actions(FALSE)
  wet$set.current.action(c(0,1,0),2)
  print(wet$get.trace())
  wet$decay()
  print(wet$get.trace())
  wet$set.current(c(0,1,0),c(0,0,1))
  print(wet$get.trace())
  wet$decay()
  print(wet$get.trace())
  wet$set.current.action(c(1,0,0),3)
  print(wet$get.trace())
  wet$decay()
  print(wet$get.trace())
}

test.cmac.trace <- function() {
  wet <- eligibility.trace.new(4, 3, lambda.decay=0.5)
  wet$set.reset.unused.actions(TRUE)
  wet$set.current.action(c(0,1,0,1),2)
  print(wet$get.trace())
  wet$decay()
  print(wet$get.trace())
  wet$set.current(c(0,1,0,1),c(0,0,1))
  print(wet$get.trace())
  wet$decay()
  print(wet$get.trace())
  wet$set.current.action(c(1,1,0,0),3)
  print(wet$get.trace())
  wet$decay()
  print(wet$get.trace())
  wet <- eligibility.trace.new(4, 3, lambda.decay=0.5)
  wet$set.reset.unused.actions(FALSE)
  wet$set.current.action(c(0,1,0,1),2)
  print(wet$get.trace())
  wet$decay()
  print(wet$get.trace())
  wet$set.current(c(0,1,0,1),c(0,0,1))
  print(wet$get.trace())
  wet$decay()
  print(wet$get.trace())
  wet$set.current.action(c(1,1,0,0),3)
  print(wet$get.trace())
  wet$decay()
  print(wet$get.trace())
}

test.cont.trace <- function() {
  wet <- eligibility.trace.new(4, 3, lambda.decay=0.5)
  wet$set.reset.unused.actions(TRUE)
  wet$set.current.action(c(0,1,0.2,0),2)
  print(wet$get.trace())
  wet$decay()
  print(wet$get.trace())
  wet$set.current(c(0,1,0.5,0),c(0,0,1))
  print(wet$get.trace())
  wet$decay()
  print(wet$get.trace())
  wet$set.current.action(c(1,0.3,0,0),1)
  print(wet$get.trace())
  wet$decay()
  print(wet$get.trace())
  wet <- eligibility.trace.new(4, 3, lambda.decay=0.5)
  wet$set.reset.unused.actions(FALSE)
  wet$set.current.action(c(0,1,0.2,0),2)
  print(wet$get.trace())
  wet$decay()
  print(wet$get.trace())
  wet$set.current(c(0,1,0.5,0),c(0,0,1))
  print(wet$get.trace())
  wet$decay()
  print(wet$get.trace())
  wet$set.current.action(c(1,0.3,0,0),1)
  print(wet$get.trace())
  wet$decay()
  print(wet$get.trace())
}

#test.discrete.trace()
#test.cmac.trace()
#test.cont.trace()

