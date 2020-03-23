mountaincar.new <- function() {
  X_INIT_MIN <- -1.2
  X_INIT_MAX <- 0.5
  V_INIT_MIN <- -0.07
  V_INIT_MAX <- 0.07
    
  X_MIN <- -1.2
  X_MAX <- 0.7
  Y_MIN <- -1
  Y_MAX <- 1.5
    
  thrust <- 0.0
  gravity <- -0.0025

  x <- 0.0
  y <- 0.0
  v <- 0.0

  reset <- function() {
    r <- runif(2)
    x <<- X_INIT_MIN + r[1]*(X_INIT_MAX - X_INIT_MIN)
    v <<- V_INIT_MIN + r[2]*(V_INIT_MAX - V_INIT_MIN)
    y <<- get.y(x)
  }

  next.x <- function(x) {
    newx <- x + v
    if ( newx < X_MIN) {
      v <<- 0.0
      newx <- X_MIN
    }
    newx;
  }

  next.v <- function(v) {
    # Calculate next value.
    nextv <- v + 0.001*thrust + gravity*cos(3*x)
    # Keep within bounds. 
    nextv <- max(V_INIT_MIN, min(V_INIT_MAX, nextv))
    nextv
  }

  get.y <- function(x) {
    sin(3.0*x)
  }
        
  step = function() {
    v <<- next.v(v)
    x <<- next.x(x)
    y <<- get.y(x)
  }

  get.action.descriptions <- function() {
    c("Left", "Zero", "Right")
  }
  
  reset() # Initialize

  # Return list of "public" functions
  list (
        reset = function() { reset() },
        get.x = function() { x },
        get.v = function() { v },
        get.y = function() { y },
        get.y.from.x = function(x) { get.y(x) },
        get.thrust = function() { thrust },
        set.x = function(value) { x <<- value },
        set.v = function(value) { v <<- value },
        set.thrust = function(value) { thrust <<- value },
        at.goal = function() { x > 0.5 },
        step = function() { step() },
        get.action.descriptions = function() { get.action.descriptions() }
        )
}

# Mountain car discretizer. Requires "Functions.R" to be run first!
mountaincardiscretizer.new <- function(xmin, xmax, xints, vmin, vmax, vints) {
  discretizer <- discrete.classifier.new(nbr.dimensions=2, minvals=c(xmin, vmin), maxvals=c(xmax,vmax), nbr.classes=c(xints,vints))

  list (
        get.class = function(x, v) {
          return(discretizer$get.index(c(x,v)))
        }
        )
}

test <- function() {
  mc <- mountaincar.new()
  mc$get.x()
  mc$get.v()
  mc$at.goal()
#for ( i in 1:1000 ) {
#  mc$step()
#  print(paste(c(mc$get.x(), mc$get.y(), mc$get.v())))
#}
  source("Functions.R")
  mcd <- mountaincardiscretizer.new(-1.2, 0.5, 4, -0.07, 0.07, 8)
  mcd$get.class(-1.2, 0.07)
}

# Mountain Car visualizer
mountaincar.visualizer.new <- function(mountain.car, r=NULL) {
  mc <- mountain.car
  has.window <- FALSE
  runner <- r
  episode <- -1
  run <- -1
  action.value.win <- 0
  car.length <- 0.4
  car.height <- 0.3
  
  update <- function() {
    # Create window if we don't have one yet
    xpoints <- c(-1.5, 0.7)
    ypoints <- c(-1.5, 1.5)
    if ( !has.window ) {
      plot(xpoints, ypoints, type = "n", xlab = "", ylab = "", axes = F,
           main = "Mountain Car")
      cat("")
      has.window <<- TRUE
    }

    # Clear previous position
    plot(xpoints, ypoints, type = "n", xlab = "", ylab = "", axes = F,
         main = "Mountain Car")
    cat("")

    # Draw terrain
    x <- seq(-1.2, 0.5, 0.1)
    y <- mc$get.y.from.x(x)
    lines(x=x,y=y,col="black")

    # Draw new position.
    car.x <- mc$get.x()
    car.y <- mc$get.y.from.x(car.x)
    polygon(c(car.x, car.x+0.5*car.length, car.x, car.x-0.5*car.length), c(car.y, car.y+0.5*car.height, car.y+car.height, car.y+0.5*car.height), col="green")

    # Labels indicating different values
    if ( !is.null(runner) )
      text(-1.5,1.5,labels=paste("Steps:", runner$get.steps()),pos=4)
    text(-1.5,1.35,labels=paste("X:", car.x),pos=4)
    text(-1.5,1.2,labels=paste("Speed:", mc$get.v()),pos=4)
    text(-1.5,1.05,labels=paste("Thrust:", mc$get.thrust()),pos=4)
    if ( !is.null(runner) ) {
      text(-1.5,0.9,labels=paste("Reward:", runner$get.reward()),pos=4)
    }
    if ( !is.null(runner) ) {
      text(-1.5,0.75,labels=paste("Episode:", runner$get.episode()),pos=4)
      text(-1.5,0.6,labels=paste("Run:", runner$get.run()),pos=4)
    }
    cat("")
  }
  
  show.action.values <- function(phi=35) {
    # Take care of having our own window. 
    old.dev <- dev.cur()
    if ( action.value.win == 0 ) {
      x11()
      par(mfrow=c(2,2))
      action.value.win <<- dev.cur()
    }
    dev.set(action.value.win)
    
    # Then plot
    fa <- runner$get.controller()$get.estimator()
    classif <- runner$get.classifier()
    cfa <- classifying.function.approximator.new(fa, classif)
    inps <- vector(mode="numeric", length=2)
    plot3d.output(cfa, show.indices=c(1,2), show.mins=c(-1.5, -0.07), show.maxs=c(0.5,0.07), show.steps=c(0.1,0.01), outindex=1, default.inputs=inps, theta=135, phi=phi, shade=0.3, xlab="X", ylab="speed", main=mc$get.action.descriptions()[1])
    plot3d.output(cfa, show.indices=c(1,2), show.mins=c(-1.5, -0.07), show.maxs=c(0.5,0.07), show.steps=c(0.1,0.01), outindex=2, default.inputs=inps, theta=135, phi=phi, shade=0.3, xlab="X", ylab="speed", main=mc$get.action.descriptions()[2])
    plot3d.output(cfa, show.indices=c(1,2), show.mins=c(-1.5, -0.07), show.maxs=c(0.5,0.07), show.steps=c(0.1,0.01), outindex=3, default.inputs=inps, theta=135, phi=phi, shade=0.3, xlab="X", ylab="speed", main=mc$get.action.descriptions()[3])
    plot(c(0,1), c(0,1), type = "n", xlab = "", ylab = "", axes = F,
         main = "Placeholder graph")
    dev.set(old.dev)
  }

  show.3d.surfaces <- function(...) {
    # Take care of having our own window. 
    old.dev <- dev.cur()
    if ( action.value.win == 0 ) {
      x11()
      par(mfrow=c(2,2))
      action.value.win <<- dev.cur()
    }
    dev.set(action.value.win)
    
    # Initialise everything
    fa <- runner$get.controller()$get.estimator()
    classif <- runner$get.classifier()
    cfa <- classifying.function.approximator.new(fa, classif)
    show.indices=c(1,2)
    show.mins=c(-1.5,-0.07)
    show.maxs=c(0.5,0.07)
    show.steps=c(0.1,0.01)
    vals <- list()
    vals[[1]] <- seq(show.mins[1], show.maxs[1], show.steps[1])
    vals[[2]] <- seq(show.mins[2], show.maxs[2], show.steps[2])

    # Then plot
    inps <- vector(mode="numeric", length=2)
    m <- create.input.matrix(indices=show.indices, mins=show.mins, maxs=show.maxs, steps=show.steps, default.inputs=inps)
    outs <- apply(m, 1, cfa$eval)
    z <- apply(outs, 2, max)
    zlim <- get.z.range(z)
    zm <- matrix(data=z, nrow=length(vals[[1]]), ncol=length(vals[[2]]), byrow=TRUE)
    persp(vals[[1]], vals[[2]], zm, zlim=zlim, theta=135, phi=35, shade=0.3, xlab="X", ylab="speed", main="Action-value function (by MAX)", ticktype="detailed", ...)
    z <- apply(outs[c(1,2),], 2, diff)
    zm <- matrix(data=z, nrow=length(vals[[1]]), ncol=length(vals[[2]]), byrow=TRUE)
    zlim <- get.z.range(zm)
    persp(vals[[1]], vals[[2]], zm, zlim=zlim, theta=135, phi=35, shade=0.3, xlab="X", ylab="speed", main="Action-value difference between actions 1/2", ticktype="detailed", ...)
    z <- apply(outs[c(2,3),], 2, diff)
    zm <- matrix(data=z, nrow=length(vals[[1]]), ncol=length(vals[[2]]), byrow=TRUE)
    zlim <- get.z.range(zm)
    persp(vals[[1]], vals[[2]], zm, zlim=zlim, theta=135, phi=35, shade=0.3, xlab="X", ylab="speed", main="Action-value difference between actions 2,3", ticktype="detailed", ...)
    z <- apply(outs[c(1,3),], 2, diff)
    zm <- matrix(data=z, nrow=length(vals[[1]]), ncol=length(vals[[2]]), byrow=TRUE)
    zlim <- get.z.range(zm)
    persp(vals[[1]], vals[[2]], zm, zlim=zlim, theta=135, phi=35, shade=0.3, xlab="X", ylab="speed", main="Action-value difference between actions 1/3", ticktype="detailed", ...)
    #contour(vals[[1]], vals[[2]], zm, xlab="angle", ylab="speed")
    dev.set(old.dev)
  }

  get.z.range <- function(z) {
    if ( identical(min(z), max(z) ) )
      zlim <- c(0,1)
    else
      zlim <- range(z, na.rm = TRUE)
    return(zlim)
  }
  
  list (
        get.mountaincar = function() { mc },
        set.mountaincar = function(mountaincar) { mc <<- mountaincar },
        get.runner = function() { runner },
        set.runner = function(r) { runner <<- r },
        update = function() { update() },
        show.action.values = function(phi=35) { show.action.values(phi) },
        show.3d.surfaces = function(...) { show.3d.surfaces(...) }
        )
}

# "Class" for running Mountain Car task.
mc.runner.new <- function(mountain.car, controller=NULL, classifier=NULL, visualiser=NULL, step.reward=0, goal.reward=1) {

  mc <- mountain.car
  c <- controller
  classif <- classifier
  v <- visualiser
  state <- vector(mode="numeric", length=2) # Default direct values
  reward <- 0
  a <- 0
  r.step <- step.reward
  r.goal <- goal.reward
  steps <- 0
  episode <- 0
  run <- 0
  
  # Go to goal. Parameters: Mountain Car object,
  # Mountain Car Discretizer object, Controller object
  # Return number of steps
  go.to.goal <- function() {
    steps <<- 0
    while ( !mc$at.goal() && steps < 150000 ) {
      state <<- classifier$get.vector(c(mc$get.x(),mc$get.v()))
      reward <<- r.step
      a <<- c$go.state.vector(state, reward)
      if ( is.na(a) || is.nan(a) ) {
        mc$set.thrust(0)
      }
      else {
        if ( a == 1 )
          mc$set.thrust(-1)
        else if ( a == 2 )
          mc$set.thrust(0)
        else
          mc$set.thrust(1)
      }
      mc$step()
      steps <<- steps + 1
      if ( !is.null(v) ) {
        v$update()
        #v$show.action.values()
      }
    }
    
    # Give reward at goal
    state <<- classifier$get.vector(c(mc$get.x(),mc$get.v()))
    reward <<- r.goal
    na <<- c$go.state.vector(state, reward)
    # Show current state if number of steps is over limit.
    if ( steps >= 150000 )
      print(paste("Maximum step limit reached, x:", mc$get.x(), ", v:",mc$get.v()))

    if ( !is.null(v) ) {
      #v$show.action.values()
      v$show.3d.surfaces()
    }

    return(steps)
  }

  # Go to goal "neps" times.
  # Other parameters: Mountain Car object,
  # Mountain Car Discretizer object, Controller object
  # Return number of steps
  run.episodes <- function(neps, run.counter=1) {
    run <<- run.counter
    cnt <- vector("integer", length=neps)
    for ( ep in 1:neps ) {
      episode <<- ep
      steps <<- go.to.goal()
      mc$reset()
      c$reset()
      cnt[ep] <- steps
    }
    return(cnt)
  }

  list (
        get.mountain.car = function() { mc },
        get.controller = function() { c },
        get.classifier = function() { classif },
        get.visualiser = function() { v },
        get.step.reward = function() { r.step },
        get.goal.reward = function() { r.goal },
        get.steps = function() { steps },
        get.episode = function() { episode },
        get.run = function() { run },
        set.mountain.car = function(mountain.car) { mc <<- mountain.car },
        set.classifier = function(classifier) { set.classifier(classifier) },
        set.controller = function(controller) { c <<- controller },
        set.visualiser = function(visualiser) { v <<- visualiser },
        set.step.reward = function(value) { r.step <<- value },
        set.goal.reward = function(value) { r.goal <<- value },
        go.to.goal = function() { go.to.goal() },
        run.episodes = function(neps, run.counter=1) { run.episodes(neps,run.counter) },
        get.reward = function() { reward }
        )
}

sarsa.mc <- function(nagents, nepisodes, classifier, lr=0.1, dr=0.9, lambda=0.9, epsilon=0.1, step.reward=0.0, goal.reward=1.0, accumulating.trace = FALSE, trace.reset=FALSE, visualize=FALSE) {

  # Create mountain car and visualiser objects
  mc <- mountaincar.new()
  if ( visualize )
    visualiser <- mountaincar.visualizer.new(mc)
  else
    visualiser <- NULL

  # Set up number of states (state variable, actually) and actions
  if ( !is.null(classifier) )
    nstates <- classifier$get.total.classes()
  else
    nstates <- 3 # If we use position, speed directly
  nactions <- 3 # 2 means clockwise/counter-cw, 3 means "no action" as third

  r <- mc.runner.new(mc,classifier=classifier,visualiser=visualiser, step.reward=step.reward, goal.reward=goal.reward)
  if ( !is.null(visualiser) )
    visualiser$set.runner(r)
  cnts <- matrix(nrow=nagents, ncol=nepisodes)
  for ( agent in 1:nagents ) {
     # Set up SARSA learner
    sl <- sarsa.learner.new(nstates, nactions, use.trace=T)
    sl$para.setup(lr, dr, lambda)
    sl$get.policy()$set.epsilon(epsilon)
    sl$get.trace()$set.accumulating.trace(accumulating.trace);
    sl$get.trace()$set.reset.unused.actions(trace.reset);
    r$set.controller(sl)
    # Run desired number of episodes
    cnts[agent,] <- r$run.episodes(nepisodes, agent)
    #print(sl$get.estimator()$get.weights())
  }
  return(cnts)
}

# Parameters: lw: Mountain Car object; mcd: discretizer; b: BIMM object
mc.slapper.new <- function(mc, mcd, b) {

  mcar <- mc
  discretizer <- mcd
  bimm <- b
  
  # Return list of "public" functions
  list (
        slap = function(state) {
          t <- mcar$get.thrust()
          if ( t < 0 )
            a <- 1
          else if ( t == 0 )
            a <- 2
          else
            a <- 3
          # SLAP if velocity and thrust go in opposite directions
          if ( mcar$get.v()*t < 0 ) {
            bimm$slap(state, a)
          }
          # Also SLAP if velocity is greater than zero but no or
          # opposite thrust. 
          else if ( mcar$get.v() > 0 && mcar$get.thrust() <= 0 ) {
            bimm$slap(state, a)
          }
        }
        )
}

bimm.mc <- function(nagents, nepisodes, classifier, k.stm=0.000001, k.ltm=1.0, slap.lr=0.1, sarsa.lr=0.1, dr=0.9, lambda=0.9, epsilon = 0.0, step.reward=0.0, goal.reward=1.0) {

  # Create mountain car object
  mc <- mountaincar.new()

  # Set up discretisation
  nstates <- classifier$get.total.classes()
  nactions <- 3

  cnts <- matrix(nrow=nagents, ncol=nepisodes)
  for ( agent in 1:nagents ) {
    # Set up SARSA learner
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
    mcslapper <- mc.slapper.new(mc, mcd, bimm)
    bimm$set.slapper(mcslapper) # Use of slapper is set up here

    # Run desired number of episodes
    cnts[agent,] <- run.episodes(nepisodes, mc, classifier, bimm, step.reward, goal.reward)
    #print(sl$get.estimator()$get.weights())
  }
  cnts
}

#test()

run.mountaincar <- function(plot=TRUE, save=FALSE, run.sarsa=TRUE, run.oiv=TRUE, run.bimm=TRUE) {
  nbr.agents <- 10
  nbr.episodes <- 100
  xints <- 8
  vints <- 8
  mins <- c(-1.2,-0.07)
  maxs <- c(0.5,0.07)
  #mcd <- discrete.classifier.new(nbr.dimensions=2, minvals=c(-1.2,-0.07), maxvals=c(0.5,0.07), nbr.classes=c(xints,vints))
  #lr <- 0.1

  #mcd <- cmac.new(nbr.layers=5, nbr.dimensions=2, minvals=mins, maxvals=maxs, nbr.classes=c(xints,vints), random.offset=TRUE)
  #lr <- 0.05

  rbfc <- rbf.classifier.new(nbrInputs=2, nbrOutputs=0, activation.function=squared.distance.activation, output.function=gaussian.output.function)
  at <- scale.translate.ranges(mins, maxs, c(0,0), c(1,1))
  rbfc$init.centroids.grid(c(-1.2,-0.07), c(0.5,0.07), c(xints,vints),affine.transformation=at)
  rbfc$set.spread(0.01)
  rbfc$set.normalize(TRUE)
  mcd <- rbfc
  lr <- 0.3
  
  # Sarsa
  if ( run.sarsa ) {
    cnt.sarsa <- sarsa.mc(nbr.agents,nbr.episodes,mcd,lr=lr,dr=0.9,lambda=0.95,epsilon=0.1,step.reward=0.0,goal.reward=1.0, accumulating.trace=FALSE, trace.reset=FALSE, visualize=TRUE)
    if ( save ) 
      write.table(cnt.sarsa,"MCres/Sarsa_MC8x8_lr01_dr09_lmda095_e01.txt", row.names=F,col.names=F)
    if (plot)
      plot(1:nbr.episodes, colMeans(cnt.sarsa), type='l', col='black', log='y')
  }
  
  # Sarsa, OIV
  if ( run.oiv ) {
    cnt.oiv <- sarsa.mc(nbr.agents,nbr.episodes,mcd,lr=lr,dr=1.0,lambda=0.9,epsilon=0.0,step.reward=-1,goal.reward=0, accumulating.trace=FALSE, trace.reset=FALSE, visualize=TRUE)
    if ( save ) 
      write.table(cnt.oiv,"MCres/OIV_MC8x8_lr01_dr1_lmda09_e0.txt", row.names=F,col.names=F)
    if (plot)
      lines(1:nbr.episodes, colMeans(cnt.oiv), type='l', col='green')
  }
  
  # BIMM
  if ( run.bimm ) {
    cnt.bimm <- bimm.mc(nbr.agents,nbr.episodes,mcd, k.stm=0.000001,k.ltm=1.0,slap.lr=1,sarsa.lr=lr,dr=0.9,lambda=0.95,epsilon=0.0,step.reward=0.0,goal.reward=1.0, accumulating.trace=FALSE, trace.reset=FALSE, visualize=TRUE)
    if ( save ) 
      write.table(cnt.bimm,"MCres/BIMM_MC8x8_stmw0000001_ltmw1_slr1_qlr01_dr09_lmda095_e0.txt", row.names=F,col.names=F)
    if (plot)
      lines(1:nbr.episodes, colMeans(cnt.bimm), type='l', col='red')
  }
  
  list (
        get.cnt.sarsa = function() { cnt.sarsa },
        get.cnt.oiv = function() { cnt.oiv },
        get.cnt.bimm = function() { cnt.bimm }
        )
}

