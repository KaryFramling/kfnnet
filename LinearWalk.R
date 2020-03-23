linearwalk.new <- function(nstates) {

  ns <- nstates # Number of states
  na <- 2 # Number of actions
  state <- floor(runif(1)*(ns - 2)) + 1 # Not in end state initially
  prev.state <- -1
  steps <- 0

  reset <- function() {
    state <<- floor(runif(1)*(ns - 2)) + 1 # Not in end state initially
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
        get.prev.state = function() { prev.state },
        get.steps = function() { steps },
        get.nbr.states = function() { ns },
        at.goal = function() { state == ns },
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

# Go to goal. Parameters: Linear Walk object,
# Controller object, step reward, goal reward
# Return number of steps
go.to.goal <- function(lw, cntrl, r.step, r.goal) {
  steps <- 0
  while ( !lw$at.goal() && steps <= 1000000 ) {
    na <- cntrl$go.state(lw$get.state(), r.step)
    lw$step(na)
    steps <- steps + 1
  }
  # Give reward at goal
  na <- cntrl$go.state(lw$get.state(), r.goal)
  steps
}

# Go to goal "neps" times.
# Other parameters: Linear Walk object, Controller object,
# step reward, goal reward.
# Return number of steps
run.episodes <- function(neps, lw, cntrl, r.step, r.goal) {
  cnt <- vector("integer", length=neps)
  for ( ep in 1:neps ) {
    steps <- go.to.goal(lw, cntrl, r.step, r.goal)
    lw$reset()
    cntrl$reset()
    cnt[ep] <- steps
    #cat(steps); cat(" ")
  }
  #cat("\n")
  cnt
}

sarsa.lw <- function(states, nagents, nepisodes, lr=0.1, dr=0.9, lambda=0.9, epsilon=0.1, step.reward=0.0, goal.reward=1.0) {

  nstates <- states
  nactions <- 2

  # Create linear walk object
  lw <- linearwalk.new(nstates)
  
  cnts <- matrix(nrow=nagents, ncol=nepisodes)
  for ( agent in 1:nagents ) {
    # Set up SARSA learner
    sl <- sarsa.learner.new(nstates, nactions)
    tr <- eligibility.trace.new(nstates, nactions)
    sl$set.trace(tr)
    sl$para.setup(lr, dr, lambda)
    sl$get.policy()$set.epsilon(epsilon)
    # Run desired number of episodes
    cnts[agent,] <- run.episodes(nepisodes, lw, sl, step.reward, goal.reward)
  }
  cnts
}

bimm.lw <- function(states, nagents, nepisodes, k.stm=0.000001, k.ltm=1.0, slap.lr=0.1, sarsa.lr=0.1, dr=0.9, lambda=0.9, epsilon = 0.0) {

  nstates <- states
  nactions <- 2

  # Create linear walk object
  lw <- linearwalk.new(nstates)
  
  cnts <- matrix(nrow=nagents, ncol=nepisodes)
  for ( agent in 1:nagents ) {
    # Set up BIMM learner
    sl <- sarsa.learner.new(nstates, nactions)
    tr <- eligibility.trace.new(nstates, nactions)
    sl$set.trace(tr)
    sl$para.setup(sarsa.lr, dr, lambda)
    
    bimm <- bimm.new(nstates, nactions)
    bimm$get.stm()$set.lr(slap.lr)
    bimm$set.stm.ar(k.stm)
    bimm$set.ltm.ar(k.ltm)
    bimm$set.ltm(sl)
    bimm$get.policy()$set.epsilon(epsilon)
    lwslapper <- lw.slapper.new(lw, bimm)
    bimm$set.slapper(lwslapper) # Use of slapper is set up here

    #sl$get.policy()$set.epsilon(0.1)
    # Run desired number of episodes
    cnts[agent,] <- run.episodes(nepisodes, lw, bimm, 0, 1)
  }
  #print(bimm$get.stm()$get.weights())
  cnts
}

run.linearwalk <- function(plot=TRUE, save=FALSE) {
  nstates <- 10
  nagents <- 10
  neps <- 100

  # Sarsa
  cnt.sarsa <- sarsa.lw(nstates,nagents,neps,lr=0.1,dr=0.9,lambda=0.9,epsilon=0.1, step.reward=0.0, goal.reward=1.0)
  if (plot)
    plot(1:neps,colMeans(cnt.sarsa),type='l')
  if ( save ) 
    write.table(cnt.sarsa,"LWres/Sarsa_LW10_lr01_dr09_lmda09_e01.txt", row.names=F,col.names=F)

  # Sarsa OIV
  cnt.oiv <- sarsa.lw(nstates,nagents,neps,lr=1.0, dr=1.0, lambda=0.9, epsilon=0.0, step.reward=-1, goal.reward=0)
  if (plot)
    lines(1:neps,colMeans(cnt.oiv),type='l', col='green')
  if ( save ) 
    write.table(cnt.oiv,"LWres/OIV_LW10_lr1_dr1_lmda09_e0.txt", row.names=F,col.names=F)
  
  # BIMM
  cnt.bimm <-bimm.lw(nstates,nagents,neps,k.stm=0.000001,k.ltm=1.0,slap.lr=1,sarsa.lr=0.1,dr=0.9,lambda=0.9,epsilon = 0.0)
  if (plot)
    lines(1:neps,colMeans(cnt.bimm),type='l', col='red')
  if ( save ) 
    write.table(cnt.bimm,"LWres/BIMM_LW10_stmw0000001_ltmw1_slr1_qlr01_dr09_lmda09_e0.txt", row.names=F,col.names=F)
    
  list (
        get.cnt.sarsa = function() { cnt.sarsa },
        get.cnt.oiv = function() { cnt.oiv },
        get.cnt.bimm = function() { cnt.bimm },
        )
}

