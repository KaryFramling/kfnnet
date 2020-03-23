continuous.grid.new <- function(xmin=0, xmax=1, ymin=0, ymax=1, xstart=0, ystart=0, xgoal=1, ygoal=1, stepsize=0.05) {

  x.min <- xmin
  x.max <- xmax
  y.min <- ymin
  y.max <- ymax
  x.start <- xstart
  y.start <- ystart
  x.goal <- xgoal
  y.goal <- ygoal
  x <- x.start
  y <- y.start
  steps <- 0
  step.size <- stepsize
  tolerance <- 0
  
  reset <- function() {
    x <<- x.start
    y <<- y.start
    steps <<- 0
  }

  at.goal <- function() {
    return ( sqrt((x.goal - x)^2 + (y.goal - y)^2) <= tolerance )
  }

  # Action can be one of four directions: 1=north, 2=east, 3=south, 4=west
  # Return true if step was possible, false if not (e.g. trying to go
  # out from area)
  step <- function(action) {
    xold <- x
    yold <- y
    if ( action == 1 )
      y <<- min(y.max, y + step.size)
    else if ( action == 2 )
      x <<- min(x.max, x + step.size)
    else if ( action == 3 )
      y <<- max(y.min, y - step.size)
    else if ( action == 4 )
      x <<- max(x.min, x - step.size)
    if ( x == xold && y == yold )
      return(FALSE)
    steps <<- steps + 1 # Only count "real" steps. 
    return(TRUE)
  }
  
  # Return list of "public" functions
  list (
        reset = function() { reset() },
        get.x = function() { x },
        get.y = function() { y },
        get.steps = function() { steps },
        at.goal = function() { at.goal() },
        step = function(action) { step(action) }
        )
}

# Mountain car discretizer. Requires "Functions.R" to be run first!
continuousgrid.discretizer.new <- function(xmin, xmax, xints, ymin, ymax, yints) {
  pxmin <- xmin
  pxmax <- xmax
  pxints <- xints
  pymin <- ymin
  pymax <- ymax
  pyints <- yints

  xdiscr <- discretizer.new(pxmin, pxmax, pxints)
  ydiscr <- discretizer.new(pymin, pymax, pyints)

  list (
        get.class = function(x, y) {
          xind <- xdiscr$get.class(x)
          yind <- ydiscr$get.class(y)
          max(1, (yind - 1)*pxints + xind)
        }
        )
}

test.continuous.grid <- function() {
  cg <- continuous.grid.new()
  while ( !cg$at.goal() ) {
    cg$step(min(4,floor(runif(1,1,5))))
  }
  print(cg$get.steps())
}

#test.continuous.grid()

# Return either state index (numeric) or vector with x,y-values
# depending on if gwd or state.vector is NULL. One of them should
# be NULL but not the other. 'cgw' is the grid world object.
get.state <- function(cgw, gwd=NULL, state.vector=NULL) {
  if ( is.null(gwd) ) {
    state[1] = cgw$get.x()
    state[2] = cgw$get.y()
  }
  else {
    state <- gwd$get.class(cgw$get.x(), cgw$get.y())
  }
  return(state)
}

# Go to goal. Parameters: Continuous Grid world object,
# grid world discretizer object, Controller object, step
# reward, goal reward. 
# Return number of steps
go.to.goal <- function(cgw, gwd=NULL, cntrl, r.step, r.goal) {
  # If we are not using a lookup-table (discretiser), then the controller wants
  # to have state variable values directly.
  if ( is.null(gwd) ) {
    state.vector <- vector(mode='numeric', length=cntrl$get.nbr.inputs())
  }

  # Run the given number of iterations to goal
  steps <- 0
  while ( !cgw$at.goal() && steps <= 1000000 ) {
    state <- get.state(cgw, gwd, state.vector)
    na <- cntrl$go.state(state, r.step)
    cgw$step(na)
    steps <- steps + 1
  }
  # Give reward at goal
  state <- get.state(cgw, gwd, state.vector)
  na <- cntrl$go.state(state, r.goal)
  steps
}

# Go to goal "neps" times.
# Other parameters: Continuous Grid world object,
# grid world discretizer object, Controller object, step
# reward, goal reward.
# Return number of steps
run.episodes <- function(neps, cgw, gwd, cntrl, r.step, r.goal) {
  cnt <- vector("integer", length=neps)
  for ( ep in 1:neps ) {
   steps <- go.to.goal(cgw, gwd, cntrl, r.step, r.goal)
   cgw$reset()
   if ( TRUE ) {
     wt <- cntrl$get.trace()$get.trace()
     m <- matrix(wt[1,],nrow=10,ncol=10)
     persp(z=m,phi=30)
   }
   cntrl$reset()
   cnt[ep] <- steps
  }
  cnt
}

sarsa.cgw.new <- function(nagents, nepisodes, xints=10, yints=10, lr=0.1, dr=0.9, lambda=0.9, epsilon=0.1, step.reward=0.0, goal.reward=1.0) {

  # Create grid world object
  cgw <- continuous.grid.new()

  # Set up discretisation
  gwd <- continuousgrid.discretizer.new(0, 1, xints, 0, 1, yints)
  nstates <- xints*yints
  nactions <- 4

  cnts <- matrix(nrow=nagents, ncol=nepisodes)
  for ( agent in 1:nagents ) {
     # Set up SARSA learner
    sl <- sarsa.learner.new(nstates, nactions, use.trace=T)
    sl$para.setup(lr, dr, lambda)
    sl$get.policy()$set.epsilon(epsilon)
    # Run desired number of episodes
    cnts[agent,] <- run.episodes(nepisodes, cgw, gwd, sl, step.reward, goal.reward)
  }

  # Return list of "public" functions
  list (
        get.sarsa.learner = function() { sl },
        get.cnts = function() { cnts },
        )
}

nrbf.cgw.new <- function(nagents, nepisodes, xkernels=2, ykernels=2, lr=0.1, dr=1.0, lambda=0.9, epsilon=0.1, step.reward=0.0, goal.reward=1.0) {

  # Create grid world object
  cgw <- continuous.grid.new()

  # Set up RBF approximator
  rbf <- rbf.new(nbrInputs=2, nbrOutputs=4, n.hidden.neurons=4, activation.function=squared.distance.activation, output.function=imqe.output.function)
  set.nrbf(TRUE) # Use Normalised RBF or not?
  
  cnts <- matrix(nrow=nagents, ncol=nepisodes)
  for ( agent in 1:nagents ) {
    # Set up SARSA learner
    sl <- sarsa.learner.new(nstates, nactions)
    sl$para.setup(lr, dr, lambda)
    sl$get.policy()$set.epsilon(epsilon)
    # Run desired number of episodes
    cnts[agent,] <- run.episodes(nepisodes, cgw, gwd, sl, step.reward, goal.reward)
  }

  # Return list of "public" functions
  list (
        get.sarsa.learner = function() { sl },
        get.cnts = function() { cnts },
        )
}

source("Functions.R")
source("Adaline.R")
source("SarsaLearner.R")
source("EligibilityTrace.R")
source("NeuralLayer.R")

keybd <- function(key) {
  cat("Key <", key, ">\n", sep = "")
}

scgw <- sarsa.cgw.new(1,1000,xints=10,yints=10,lr=0.1,dr=0.9,lambda=0.95,epsilon=0.1,step.reward=0.0,goal.reward=1.0)
#print(colMeans(scgw$get.cnts()))
weights <- scgw$get.sarsa.learner()$get.estimator()$get.weights()
#m <- matrix(weights[1,],nrow=10,ncol=10)
i <- array(c(max.col(t(weights)),1:ncol(weights)),dim=c(ncol(weights),2))
m <- matrix(weights[i],nrow=10,ncol=10)
#contour(z=m)
persp(z=m, shade=0.6)
#getGraphicsEvent("Press enter for trace", onKeybd=keybd)
