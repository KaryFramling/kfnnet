grid.new <- function(width, height, xstart, ystart, xgoal, ygoal) {

  w <- width
  h <- height
  x.start <- xstart
  y.start <- ystart
  x.goal <- xgoal
  y.goal <- ygoal
  x <- x.start
  y <- y.start
  steps <- 0
  random.transition.rate <- 0.0
  
  reset <- function() {
    x <<- x.start
    y <<- y.start
    steps <<- 0
  }

  at.goal <- function() {
    return ( x == x.goal && y == y.goal )
  }

  # Action can be one of four directions: 1=north, 2=east, 3=south, 4=west
  # Return true if step was possible, false if not (e.g. trying to go
  # out from area)
  step <- function(action) {
    xold <- x
    yold <- y
    if ( random.transition.rate > 0.0 ) {
      if ( runif(1) < random.transition.rate )
        action <- min(4,floor(runif(1,1,5)))
    }
    if ( action == 1 )
      y <<- min(height, y + 1)
    else if ( action == 2 )
      x <<- min(width, x + 1)
    else if ( action == 3 )
      y <<- max(1, y - 1)
    else if ( action == 4 )
      x <<- max(1, x - 1)
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
        get.state = function() { (y - 1)*width + x },
        get.steps = function() { steps },
        get.random.transition.rate = function() { random.transition.rate },
        set.random.transition.rate = function(value) { random.transition.rate <<- value },
        at.goal = function() { at.goal() },
        step = function(action) { step(action) }
        )
}

# Go to goal. Parameters: Grid object,
# Controller object, step reward, goal reward
# Return number of steps
go.to.goal <- function(world, cntrl, r.step, r.goal) {
  steps <- 0
  while ( !world$at.goal() && steps <= 1000000 ) {
    na <- cntrl$go.state(world$get.state(), r.step)
    world$step(na)
    steps <- steps + 1
  }
  # Give reward at goal
  na <- cntrl$go.state(world$get.state(), r.goal)
  steps
}

# Go to goal "neps" times.
# Other parameters: Grid object, Controller object,
# step reward, goal reward.
# Return number of steps
run.episodes <- function(neps, world, cntrl, r.step, r.goal) {
  cnt <- vector("integer", length=neps)
  for ( ep in 1:neps ) {
    steps <- go.to.goal(world, cntrl, r.step, r.goal)
    world$reset()
    cntrl$reset()
    cnt[ep] <- steps
  }
  cnt
}

sarsa.grid <- function(width, height, nagents, nepisodes, lr=0.1, dr=0.9, lambda=0.9, epsilon=0.1, random.transition.rate=0.0, step.reward=0.0, goal.reward=1.0) {

  nstates <- width*height
  nactions <- 4

  # Create linear walk object with start in lower left corner, goal in
  # upper right corner. 
  grid <- grid.new(width, height, 1, 1, width, height)
  grid$set.random.transition.rate(random.transition.rate)
  
  cnts <- matrix(nrow=nagents, ncol=nepisodes)
  for ( agent in 1:nagents ) {
    # Set up SARSA learner
    sl <- sarsa.learner.new(nstates, nactions)
    tr <- eligibility.trace.new(nstates, nactions)
    sl$set.trace(tr)
    sl$para.setup(lr, dr, lambda)
    sl$get.policy()$set.epsilon(epsilon)
    # Run desired number of episodes
    cnts[agent,] <- run.episodes(nepisodes, grid, sl, step.reward, goal.reward)
  }
  cnts
}

test.grid <- function() {
  cg <- grid.new(20, 20, 1, 1, 10, 10)
  while ( !cg$at.goal() ) {
    cg$step(min(4,floor(runif(1,1,5))))
  }
  print(cg$get.steps())
}

#test.grid()
