# Implementation of the random walk task in Singh&Sutton(1996)
# "Reinforcement Learning with Replacing Eligibility Traces".
# Here we have end states in both ends, left end with -1 reward
# and right end with +1 reward. 
randomwalk.new <- function(nstates) {

  ns <- nstates # Number of states
  na <- 2 # Number of actions
  state <- min(ns,floor(runif(1)*(ns - 2)) + 2) # Not in end state initially
  prev.state <- -1
  steps <- 0
  init.state <- -1 # -1 means no fixed init state

  reset <- function() {
    if ( init.state == -1 )
      state <<- min(ns,floor(runif(1)*(ns - 2)) + 2) # Random state
    else
      state <<- init.state
    prev.state <<- -1
    steps <<- 0
  }

  # Go one step in the direction indicated by "action".
  # 1: go left; 2: go right.
  # Return new state
  step <- function(action) {
    prev.state <<- state
    if ( action == 1 ) { # backward
      if ( state > 1 )
        state <<- state - 1
    }
    if ( action == 2 ) { # forward
      if ( state < ns )
        state <<- state + 1
    }
    state
  }
  
  # Return list of "public" functions
  list (
        reset = function() { reset() },
        get.state = function() { state },
        get.init.state = function() { init.state },
        set.init.state = function(s) { init.state <<- s },
        get.prev.state = function() { prev.state },
        get.steps = function() { steps },
        get.nbr.states = function() { ns },
        at.goal = function() { (state == ns) | (state == 1) },
        step = function(action) { step(action) }
        )
}

# Parameters: lw: linearwalk object; b: BIMM object
lw.slapper.new <- function(lw, b) {

  linearwalk <- lw
  bimm <- b
  
  # Return list of "public" functions
  list (
        slap = function(state) {
          ps <- lw$get.prev.state()
          s <- lw$get.state()
          if ( ps != -1 ) {
            if ( s - ps >= 0 ) {
              bimm$slap(state, 1) # Do not go back to left
            }
            else if ( s - ps < 0 ) {
              bimm$slap(state, 2) # Do not go back to right
            }
          }
        }
        )
}

# Go to goal. Parameters: Random Walk object,
# Controller object, step reward, goal reward
# Return number of steps
go.to.goal <- function(rw, cntrl, r.step = 0, r.leftgoal = -1, r.rightgoal = 1) {
  steps <- 0
  while ( !rw$at.goal() && steps <= 1000000 ) {
    na <- cntrl$go.state(rw$get.state(), r.step)
    rw$step(na)
    steps <- steps + 1
  }
  # Give reward at goal
  if ( rw$get.state() == 1 )
    r <-  r.leftgoal
  else
    r <-  r.rightgoal
  na <- cntrl$go.state(rw$get.state(), r)
  return(steps)
}

# Go to goal "neps" times.
# Other parameters: Random Walk object, Controller object,
# step reward, goal reward.
# Return number of steps
run.episodes <- function(neps, rw, cntrl, r.step = 0, r.leftgoal = -1, r.rightgoal = 1) {
  cnt <- vector("integer", length=neps)
  for ( ep in 1:neps ) {
    steps <- go.to.goal(rw, cntrl, r.step, r.leftgoal, r.rightgoal)
    rw$reset()
    cntrl$reset()
    cnt[ep] <- steps
    #cat(steps); cat(" ")
  }
  #cat("\n")
  cnt
}

sarsa.rw <- function(states = 21, init.state = 11, nagents, nepisodes, lr=0.1, dr=0.9, lambda=0.9, epsilon=0.1, r.step = 0, r.leftgoal = -1, r.rightgoal = 1) {

  nstates <- states
  nactions <- 2

  # Create linear walk object
  rw <- randomwalk.new(nstates)
  rw$set.init.state(init.state)
  
  cnts <- matrix(nrow=nagents, ncol=nepisodes)
  for ( agent in 1:nagents ) {
    # Set up SARSA learner
    sl <- sarsa.learner.new(nstates, nactions, use.trace=T)
    sl$para.setup(lr, dr, lambda)
    sl$get.policy()$set.epsilon(epsilon)
    # Run desired number of episodes
    cnts[agent,] <- run.episodes(nepisodes, rw, sl, r.step = 0, r.leftgoal = -1, r.rightgoal)
    print(sl$get.estimator()$get.weights());
  }
  return(cnts)
}

run.randomwalk <- function(plot=TRUE, save=FALSE) {
  source("Functions.R")
  source("NeuralLayer.R")
  source("Adaline.R")
  source("SarsaLearner.R")
  source("BIMM.R")
  source("EligibilityTrace.R")

  nstates <- 21
  nagents <- 10
  neps <- 10
  initstate <- 11
  
  # Sarsa
  cnt.sarsa <- sarsa.rw(nstates,initstate,nagents,neps,lr=0.5,dr=1.0,lambda=0.9,epsilon=0.0, r.step = 0, r.leftgoal = -1, r.rightgoal = 1)
  if (plot)
    plot(1:neps,colMeans(cnt.sarsa),type='l')
  if ( save ) 
    write.table(cnt.sarsa,"LWres/Sarsa_LW10_lr01_dr09_lmda09_e01.txt", row.names=F,col.names=F)
    
  list (
        get.cnt.sarsa = function() { cnt.sarsa }
        )
}


