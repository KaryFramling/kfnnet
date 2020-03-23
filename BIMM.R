bimm.new <- function(nbrStateVariables, nbrActions) {

  ns <- nbrStateVariables
  na <- nbrActions
  
  stm.init.min <- 0.0
  stm.init.max <- 1.0  
  stm <- adaline.new(ns, na)
  #Initialize STM here
  w <- runif(ns*na, min=stm.init.min, max=stm.init.max)
  stm$set.weights(matrix(w, nrow = na, ncol = ns))
  stm$set.lr(0.1)
  stm.ar <- 0.000001 # Activation ratios between STM and LTM
  stm$set.mixtw(stm.ar)

  ltm <- sarsa.learner.new(ns, na)
  ltm.ar <- 1.0
  ltm$para.setup(0.1, 0.9, 0.95)
  ltm$get.policy()$set.epsilon(0.0)

  slap.marg <- 0.1
  shap.marg <- 1.0

  policy <- e.greedy.new(0.0) # Use e-greedy by default
  
  input <- vector("integer", length=ns)
  out <- c()
  tvals <- c()

  action <- -1
  
  slapper <- NULL
  
  reset <- function() {
    #Reset STM&LTM here!
    w <- runif(ns*na, min=stm.init.min, max=stm.init.max)
    stm$set.weights(matrix(w, nrow = na, ncol = ns))
    ltm$reset()
  }

  set.stm.ar <- function(kstm) {
    stm.ar <<- kstm
    stm$set.mixtw(stm.ar)
  }
  
  # Shortcut method to set up most common parameters
  para.setup <- function(slrate, llrate, drate, kstm, kltm, lambda) {
    set.stm.ar(kstm)
    ltm.ar <<- kltm
    if ( !is.null(stm) )
      stm$set.lr(slrate)
    if ( !is.null(ltm) ) {
      ltm$set.lr(llrate)
      ltm$set.dr(drate)
      ltm$set.lambda(lambda)
    }
  }

  get.outputs <- function(state) {
    input[] <<- 0
    input[state] <<- 1
    out <<- ltm.ar*ltm$get.outputs(state) + stm.ar*stm$eval(input)
    out
  }
  
  # Return index of greedy action for state "state"
  get.greedy <- function(state) {
    random.of.max(get.outputs(state))
  }

  # "next.state" should be set correctly before calling this
  get.action <- function(state) {
    # Evaluate for "state"
    out <<- get.outputs(state)

    # Decide here on policy: eps-greedy, Boltzmann, other
    action <<- policy$get.action(out)
    action
  }

  # SLAP given state/action. 
  # Since we only want to train one action (output), we set
  # the other target values to be the same as the output values. 
  slap <- function(state, action) {
    # Evaluate for "state"
    out <<- get.outputs(state)
    min.activation <- min(out) - slap.marg
    tvals <<- out
    tvals[action] <<- min.activation
    stm$train(tvals)
  }
  
  # SHAP given state/action. 
  # Since we only want to train one action (output), we set
  # the other target values to be the same as the output values. 
  shap <- function(state, action) {
    # Evaluate for "state"
    out <<- get.outputs(state)
    max.activation <- max(out) + shap.marg
    tvals <<- out
    tvals[action] <<- max.activation
    stm$train(tvals)
  }
  
  # Return list of "public methods"
  list (
        reset = function() { reset() },
        get.dr = function() { ltm$get.dr() },
        set.dr = function(value) { ltm$set.dr(value) },
        get.curr.state = function() { curr.state },
        get.trace = function () { ltm$get.trace },
        set.trace = function (t) { ltm$set.trace(t) },
        get.policy = function () { policy },
        set.policy = function (p) { policy <<- p },
        get.stm = function () { stm },
        set.stm = function(o) { stm <<- o },
        get.ltm = function () { ltm },
        set.ltm = function(o) { ltm <<- o },
        get.stm.ar = function() { stm.ar },
        set.stm.ar = function(v) { stm.ar <<- v },
        get.ltm.ar = function() { ltm.ar },
        set.ltm.ar = function(v) { ltm.ar <<- v },
        slap = function(state, action) { slap(state, action) },
        shap = function(state, action) { shap(state, action) },
        get.action = function() { action },
        get.slapper = function() { slapper },
        set.slapper = function(o) { slapper <<- o },
        
        go.state = function(state, reward) {
          # SLAP/SHAP if needed
          if ( !is.null(slapper) ) {
            slapper$slap(state) # Also SHAP is applicable
          }
          # Get action for new state
          act <- get.action(state)
          # Move Q-learner to new state
          ltm$go.state.action(state, act, reward)
          # Return next action to take
          act
        },
        
        get.greedy = function() { get.greedy(curr.state) },

        get.outputs = function(state) { get.outputs(state) },

        para.setup = function(slrate, llrate, drate, kstm, kltm, lambda) {
          para.setup(slrate, llrate, drate, kstm, kltm, lambda) 
        }
        )
}

