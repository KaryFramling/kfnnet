# Discrete-version Sarsa learner
sarsa.learner.new <- function(nbrStateVariables, nbrActions, use.trace=FALSE) {

  ns <- nbrStateVariables
  na <- nbrActions
  
  prev.state <- -1
  curr.state <- -1
  next.state <- -1
  curr.action <- -1
  next.action <- -1
  a <- 0 # Q-value of "last taken action"
  a1 <- 0 # Q-value of "action to take in new state"

  dr <- 1.0 # Discount rate

  policy <- e.greedy.new(0.1) # Use e-greedy by default
  
  input <- vector("integer", length=ns)
  out <- c()
  tvals <- c()
  
  estimator <- adaline.new(nbrStateVariables, nbrActions, use.trace)
  
  # Reset to new "initial" state, i.e. forget state information.
  # This keeps the learnt model.
  reset <- function() {
    prev.state <<- -1
    curr.state <<- -1
    next.state <<- -1
    curr.action <<- -1
    next.action <<- -1
    tr <- estimator$get.trace()
    if ( !is.null(tr) )
      tr$reset()
  }

  # Set discount rate, inform trace about change also (if trace is used)
  set.dr <- function(value) {
    dr <<- value
    tr <- estimator$get.trace()
    if (!is.null(tr))
      tr$set.dr(dr)
  }

  # Return index of greedy action for state "state"
  get.outputs <- function(state) {
    input[] <<- 0
    input[state] <<- 1
    estimator$eval(input)
  }

  # Return index of greedy action for state "state"
  get.greedy <- function(state) {
    random.of.max(get.outputs(state))
  }

  # "next.state" should be set correctly before calling this
  get.next.action <- function() {
    # Evaluate for "next.state"
    input[] <<- 0
    input[next.state] <<- 1
    out <<- estimator$eval(input)

    # Decide here on policy: eps-greedy, Boltzmann, other
    next.action <<- policy$get.action(out)
    a1 <<- out[next.action]
    next.action
  }

  # Before calling this function, we need to have the following
  # values set correctly:
  # - curr.state: state we took last action for
  # - curr.action: last action taken
  # - a: action-value of last action taken
  # - next.state: state for which reward is received
  # - a1: action-value for action that we will take next
  give.reward <- function(r) {
    # Set input values for "curr.state" and evaluate.
    if ( curr.state != -1 ) {
      input[] <<- 0
      input[curr.state] <<- 1
      out <<- estimator$eval(input)
      delta <- r + dr*a1 - out[curr.action]

      # Special treatment when eligibility trace is used
      tr <- estimator$get.trace()
      if ( is.null(tr) ) {
        out[] <<- 0
        out[curr.action] <<- delta
        estimator$train.with.delta(out)
      }
      else {
        #update.trace(curr.state, curr.action)
        estimator$train.with.delta(delta)
      }
    }
  }

  get.lambda <- function() { 
    tr <- estimator$get.trace()
    if (!is.null(tr))
      tr$get.lambda()
    else
      0
  }

  set.lambda <- function(l) { 
    tr <- estimator$get.trace()
    if (!is.null(tr))
      tr$set.lambda(l)
  }

  # Given action taken in given state - update eligibility trace.
  # Should this be moved somewhere else, e.g. Adaline "train"?
  update.trace <- function(state, action) {
    tr <- estimator$get.trace()
    if ( !is.null(tr) ) {
      tr$set.current.lookup.table(state, action)
    }
  }

  # Shortcut method to set up most common parameters
  para.setup <- function(lrate, drate, lambda) {
    if ( !is.null(estimator) )
      estimator$set.lr(lrate)
    set.dr(drate)
    set.lambda(lambda)
  }
  
  # Go to new state where given action will be the next one
  # taken and give reward. This method is special case for
  # BIMM, for instance, where the action to take depends
  # on external decisions. 
  go.state.action <- function(state, action, reward) {
    # Evaluate for "next.state"
    next.state <<- state
    input[] <<- 0
    input[next.state] <<- 1
    out <<- estimator$eval(input)
    next.action <<- action
    a1 <<- out[next.action]
    # Update trace if it is used
    if ( curr.state != -1 )
      update.trace(curr.state, curr.action)
    # Train (give.reward)
    give.reward(reward)
    # Store needed information for future
    prev.state <<- curr.state
    curr.state <<- next.state
    curr.action <<- next.action
    a <<- a1
    # Return action to take in this state
    next.action
  }
        
  # Go to new state, decide on next action and give reward
  go.state <- function(state, reward) {
    next.state <<- state
    # Decide on next action, get "next.action" and "a1"
    get.next.action()
    # Update trace if it is used
    if ( curr.state != -1 )
      update.trace(curr.state, curr.action)
    # Train (give.reward)
    give.reward(reward)
    # Store needed information for future
    prev.state <<- curr.state
    curr.state <<- next.state
    curr.action <<- next.action
    a <<- a1
    # Return action to take in this state
    next.action
  }
  
  # Return list of "public methods"
  list (
        get.nbr.inputs = function() { ns },
        get.nbr.outputs = function() { na },
        reset = function() { reset() },
        get.dr = function() { dr },
        set.dr = function(value) { set.dr(value) },
        get.curr.state = function() { curr.state },
        get.trace = function () { estimator$get.trace() },
        set.trace = function (t) { estimator$set.trace(t); t$set.dr(dr) },
        get.policy = function () { policy },
        set.policy = function (p) { policy <<- p },
        get.estimator = function () { estimator },
        set.estimator = function (e) { estimator <<- e },
        get.lambda = function() { get.lambda() },
        set.lambda = function(l) { set.lambda(l) },

        # Go to new state where given action will be the next one
        # taken and give reward. This method is special case for
        # BIMM, for instance, where the action to take depends
        # on external decisions. 
        go.state.action = function(state, action, reward) {
          go.state.action(state, action, reward)
        },
        
        # Go to new state, decide on next action and give reward
        go.state = function(state, reward) { go.state(state, reward) },
        get.greedy = function() { get.greedy(curr.state) },
        get.outputs = function(state) { get.outputs(state) },
        para.setup = function(lrate, drate, lambda) {
          para.setup(lrate, drate, lambda) 
        }
        )

}

