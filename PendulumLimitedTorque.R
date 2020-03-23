# Pendulum swing up task with limited torque formulas taken
# from Schaal, 1997 but should be identical with both older
# and newer sources.
#
# Created 14.8.2006, Kary Främling
pendulum.limited.torque.new <- function() {

  g <- 9.81 # Gravity
  mu <- 0.01 # Friction. Doya uses 0.01, Schaal 0.05. Error in Schaal?
  tau.max <- 5.0 # % Nm max torque by default

  m <- 1 # Mass of pendulum "head"
  l <- 1 # Length of pendulum
  trial.length <- 60 # 60 seconds total trial time (45 seconds uptime is cosidered good)
  up.angle.limit <- pi/2 # angle smaller than +/-pi/2 is considered as being "up"

  time <- 0.0 # Current time
  time.step <- 0.02 # Seconds. Runge-Kutta or similar should be used for more exact results.
  
  # Bogus values, real ones are set in "reset" method.
  theta <- 0.0 # Pendulum angle, radians. Initially random value from [-PI,PI].
  dtheta <- 0.0 # Pendulum angular speed, radians/s. Initial value zero.
  ddtheta <- 0.0 # Angular acceleration, radians/s^2. 
  tau <- 0.0 # Applied torque

  reset <- function() {
    r <- runif(1)
    theta <<- -pi + r*2.0*pi
    #theta <<- 0
    dtheta <<- 0.0
    ddtheta <<- 0.0
    tau <<- 0.0
    time <<- 0.0
  }

  next.theta <- function(angle, speed) {
    res <- angle + speed*time.step;
    if ( res > pi )
      res <- -2*pi + res
    else if ( res <= -pi )
      res <- 2*pi + res
    return(res)
  }

  next.dtheta <- function(speed, acc) {
    res <- speed + acc*time.step
    return(res)
  }

  # Get new acceleration after one time step. 
  next.ddtheta <- function(torque, angle, speed, acc) {
    sqterm = m*l*l;
    res <- (g*sin(angle)/l - mu*speed/sqterm + torque/sqterm)*time.step
    return(res)
  }

  step = function() {
    time <<- time + time.step
    ddtheta <<- next.ddtheta(tau, theta, dtheta, ddtheta)
    dtheta <<- next.dtheta(dtheta, ddtheta)
    theta <<- next.theta(theta, dtheta)
  }

  get.action.descriptions <- function() {
    c("Clockwise", "Anti-clockwise")
  }
  
  reset() # Initialize

  # Return list of "public" functions
  list (
        reset = function() { reset() },
        get.time = function() { time },
        get.time.step = function() { time.step },
        get.angle = function() { theta },
        get.angle.speed = function() { dtheta },
        get.angle.acc = function() { ddtheta },
        get.torque = function() { tau },
        set.time.step = function(value) { time.step <<- value },
        set.angle = function(value) { theta <<- value },
        set.angle.speed = function(value) { dtheta <<- value },
        set.torque = function(value) { tau <<- value },
        at.goal = function() { abs(theta) <= up.angle.limit },
        step = function() { step() },
        get.action.descriptions = function() { get.action.descriptions() }
        )
}

# Pendulum input space discretizer. Requires "Functions.R" to be run first!
pendulumclassifier.new <- function(rbf=FALSE, cmac=FALSE, angle.min=-pi, angle.max=pi, angle.ints=8, speed.min=-1.5, speed.max=1.5, speed.ints=8, nbr.layers=5) {
  
  dims <- 2
  mins <- c(angle.min, speed.min)
  maxs <- c(angle.max, speed.max)
  ncls <- c(angle.ints,speed.ints)

  # Calculate interval size for angles, needed at least for CMAC
  # wrap-around at +/-pi
  angle.range <- angle.max - angle.min
  angle.interval <- angle.range/angle.ints

  if ( cmac ) {
    # Special treatment for handling angle wrap. 
    discretizer  <- cmac.new(nbr.layers=nbr.layers, nbr.dimensions=dims, minvals=mins, maxvals=c(angle.max-angle.interval, speed.max), nbr.classes=c(angle.ints-1,speed.ints), random.offset=TRUE, impose.limits=c(TRUE, FALSE))
  }
  else if ( rbf ) {
    rbfc <- rbf.classifier.new(nbrInputs=dims, nbrOutputs=0, 
                               activation.function=squared.distance.activation,
                               output.function=gaussian.output.function)
    at <- scale.translate.ranges(mins, maxs, c(0,0), c(1,1))
    rbfc$init.centroids.grid(mins, maxs, ncls, affine.transformation=at)
    rbfc$set.wrapped.inputs(c(1))
    #rbfc$set.spread(1/max(ncls))
    rbfc$set.spread(0.01)
    rbfc$set.normalize(TRUE)
    discretizer <- rbfc
  }
  else {
    discretizer <- discrete.classifier.new(nbr.dimensions=dims, minvals=mins, maxvals=maxs, nbr.classes=ncls)
  }

  # We need special treatment here for CMAC and RBF due to the
  # wrap-around at +/-pi
  get.vector <- function(state) {
    if ( inherits(discretizer, "CMAC") ) {
      res <- discretizer$get.vector(state)
      # Check if angle is in "wrap-sensitive" area. In that case, we also need
      # to activate "wrapped-around" CMAC features. We only need to check this
      # on one side of the +/- limt due to how the CMAC is implemented
      # and created.
      angle <- state[1]
      if ( angle >= pi - angle.interval ) {
        #print(paste("In WRAP area, res: ", sum(res)))
        res <- as.numeric(res | discretizer$get.vector(c(angle-angle.range,state[2])))
      }
    }
    #else if ( inherits(discretizer, "RBFclassifier") ) {
      # Check if angle is in "wrap-sensitive" area. If so, we also need
      # to activate "wrapped-around" RBF nodes features. 
    #  angle <- state[1]

      # Take wrap-around into account in both directions 
    #  res <- pmax(res, discretizer$get.vector(c(angle-angle.range,state[2])))
    #  res <- pmax(res, discretizer$get.vector(c(angle+angle.range,state[2])))
    #}
    else {
      res <- discretizer$get.vector(state)
    }
    #print(sum(res))
    return(res)
  }
  
  list (
        get.total.classes = function() { discretizer$get.total.classes() },
        get.vector = function(state) { get.vector(state) },
        get.class = function(angle, speed) { discretizer$get.index(angle, speed) }
        )
}

# Pendulum visualizer
pendulum.visualizer.new <- function(pendulum, r=NULL) {
  p <- pendulum
  has.window <- FALSE
  old.angle <- 0 # Causes useless first clearing but that should be no problem
  runner <- r
  episode <- -1
  run <- -1
  action.value.win <- 0
  
  update <- function() {
    # Create window if we don't have one yet
    xpoints <- c(-1.5, 1.5)
    ypoints <- xpoints
    if ( !has.window ) {
      plot(xpoints, ypoints, type = "n", xlab = "", ylab = "", axes = F,
           main = "Pendulum with Limited Torque")
      cat("")
      has.window <<- TRUE
    }

    # Clear previous position
    plot(xpoints, ypoints, type = "n", xlab = "", ylab = "", axes = F,
         main = "Pendulum with Limited Torque")
    cat("")

    # Draw new position
    a <- pendulum$get.angle()
    lines(x=c(0,sin(a)),y=c(0,cos(a)),col="black")
    points(x=sin(a),y=cos(a),col="black",pch=19)
    cat("")
    old.angle <<- a

    # Labels indicating different values
    text(-1.4,1.5,labels=paste("Time:", p$get.time()),pos=4)
    text(-1.4,1.35,labels=paste("Angle:", a),pos=4)
    text(-1.4,1.2,labels=paste("Speed:", p$get.angle.speed()),pos=4)
    text(-1.4,1.05,labels=paste("Torque:", p$get.torque()),pos=4)
    if ( !is.null(runner) ) {
      text(-1.4,0.9,labels=paste("Reward:", runner$get.reward()),pos=4)
    }
    if ( episode > 0 )
      text(-1.4,0.75,labels=paste("Episode:", episode),pos=4)
    if ( run > 0 )
      text(-1.4,0.6,labels=paste("Run:", run),pos=4)
  }
  
  show.action.values <- function(phi=35) {
    # Take care of having our own window. 
    old.dev <- dev.cur()
    if ( action.value.win == 0 ) {
      x11()
      par(mfcol=c(1,2))
      action.value.win <<- dev.cur()
    }
    dev.set(action.value.win)
    
    # Then plot
    fa <- runner$get.controller()$get.estimator()
    classif <- runner$get.classifier()
    cfa <- classifying.function.approximator.new(fa, classif)
    inps <- vector(mode="numeric", length=2)
    plot3d.output(cfa, show.indices=c(1,2), show.mins=c(-pi,-1.5), show.maxs=c(pi,1.5), show.steps=c(pi/10,0.15), outindex=1, default.inputs=inps, theta=135, phi=phi, shade=0.3, xlab="angle", ylab="speed")
    plot3d.output(cfa, show.indices=c(1,2), show.mins=c(-pi,-1.5), show.maxs=c(pi,1.5), show.steps=c(pi/10,0.15), outindex=2, default.inputs=inps, theta=135, phi=phi, shade=0.3, xlab="angle", ylab="speed")
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
    show.mins=c(-pi,-1.5)
    show.maxs=c(pi,1.5)
    show.steps=c(pi/10,0.15)
    vals <- list()
    vals[[1]] <- seq(show.mins[1], show.maxs[1], show.steps[1])
    vals[[2]] <- seq(show.mins[2], show.maxs[2], show.steps[2])

    # Then plot
    inps <- vector(mode="numeric", length=2)
    m <- create.input.matrix(indices=show.indices, mins=show.mins, maxs=show.maxs, steps=show.steps, default.inputs=inps)
    outs <- apply(m, 1, cfa$eval)
    z <- outs[1,]
    zm <- matrix(data=z, nrow=length(vals[[1]]), ncol=length(vals[[2]]), byrow=TRUE)
    zlim <- get.z.range(zm)
    persp(vals[[1]], vals[[2]], zm, zlim=zlim, theta=135, phi=35, shade=0.3, xlab="angle", ylab="speed", main=p$get.action.descriptions()[1], ticktype="detailed", ...)
    z <- outs[2,]
    zlim <- get.z.range(z)
    zm <- matrix(data=z, nrow=length(vals[[1]]), ncol=length(vals[[2]]), byrow=TRUE)
    persp(vals[[1]], vals[[2]], zm, zlim=zlim, theta=135, phi=35, shade=0.3, xlab="angle", ylab="speed", main=p$get.action.descriptions()[2], ticktype="detailed", ...)
    z <- apply(outs, 2, max)
    zlim <- get.z.range(z)
    zm <- matrix(data=z, nrow=length(vals[[1]]), ncol=length(vals[[2]]), byrow=TRUE)
    persp(vals[[1]], vals[[2]], zm, zlim=zlim, theta=135, phi=35, shade=0.3, xlab="angle", ylab="speed", zlab="max(action-value)", main="Action-value function (by MAX)", ticktype="detailed", ...)
    z <- apply(outs, 2, diff)
    zlim <- get.z.range(z)
    zm <- matrix(data=z, nrow=length(vals[[1]]), ncol=length(vals[[2]]), byrow=TRUE)
    persp(vals[[1]], vals[[2]], zm, zlim=zlim, theta=135, phi=35, shade=0.3, xlab="angle", ylab="speed", main="Action-value difference between actions", ticktype="detailed", ...)
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
        get.pendulum = function() { p },
        set.pendulum = function(pendulum) { p <<- pendulum },
        get.runner = function() { runner },
        set.runner = function(r) { runner <<- r },
        get.episode = function() { episode },
        set.episode = function(e) { episode <<- e },
        get.run = function() { run },
        set.run = function(r) { run <<- r },
        update = function() { update() },
        show.action.values = function(phi=35) { show.action.values(phi) },
        show.3d.surfaces = function(...) { show.3d.surfaces(...) }
        )
}

# Handcoded controller for pendulum task. Implements "Controller" interface. 
handcoded.pendulum.controller.new <- function(pendulum, max.torque=5) {

  p <- pendulum
  state <- NULL
  action <- NULL
  
  set.state.vector <- function(inputs, reward) {
    state <<- inputs
    angle <- state[1]
    speed <- state[2]
    action <<- 1
    axis.force <- abs(sin(angle/6))*9.81
    nbr.actions <- 2 # Include "no torque" and maybe other actions or not
    if ( abs(angle) < pi/4 ) {
      if ( speed > 0.01 )
        action <<- 2
      else if ( speed < -0.01 )
        action <<- 1
      else
        if ( angle < -0.001 )
          action <<- 1
        else if ( angle > 0.001 )
          action <<- 2
        else
          action <<- 3
    }
    else {
      if ( speed > 0 && speed < 0.45 )
        action <<- 1
      else if ( speed < 0 && speed > -0.45 )
        action <<- 2
      else
        action <<- 3
    }

    # Truncate "unused" actions by assigning random action instead.
    if ( action > nbr.actions )
      action <<- min(floor(1 + runif(1)*(nbr.actions)), nbr.actions)
    return(c(action))
  }

  get.action.descriptions <- function() {
    c("Turn clockwise", "Turn anti-clockwise", "No torque")
  }
  
  # Return list of "public" methods. Also set "class" attribute.
  pub <- list(
              # "Controller" implementation here
              reset = function() { NULL },
              get.state = function() { state },
              set.state.vector = function(inputs, reward=0) {
                set.state.vector(inputs, reward)
              },
              get.actions = function() { c(action) },
              has.discrete.actions = function() { TRUE },
              get.action.descriptions = function() { get.action.descriptions() }
              )

  # We implement "Controller" interface
  c <- controller.new()

  class(pub) <- c("HandCodedPendulumController",class(c))
  return(pub)
}

# "Class" for running Pendulum with Limited Torque task.
plt.runner.new <- function(pendulum, controller=NULL, classifier=NULL, visualiser=NULL) {
  p <- pendulum
  c <- controller
  classif <- classifier
  v <- visualiser
  state <- vector(mode = "integer", length=2) # Default direct values
  at.goal.previous.step <- FALSE
  reward <- 0
  control.step.interval <- 1 # Change control action every N steps. This is buggy for the moment, don't use (reward given too late!)
  a <- 0
  
  # Runs for given time (in seconds). Returns number of time steps "at goal". 
  run.time <- function(time) {
    steps <- ceiling(time/p$get.time.step())
    goal.steps <- 0
    for ( i in 1:steps ) {
      if ( p$at.goal() ) 
        goal.steps <- goal.steps + 1
      angle <-p$get.angle()
      speed <- p$get.angle.speed()
      state <- c(angle, speed)
      if ( !is.null(classif) )
        state <- classif$get.vector(state)

      if ( !is.null(c) ) {
        #reward <- cos(state[1]) # Cosinus of angle reward.
        # Give reward depending on if we enter or exit goal zone
        # plus some reward for other situations.
        reward.limit <- pi/10
        if ( i > 1 && abs(angle) < reward.limit && !at.goal.previous.step ) 
          reward <<- 10
        else if ( i > 1 && abs(angle) > reward.limit && at.goal.previous.step )
          reward <<- -10
        else 
          #reward <<- cos(angle) - 1
          reward <<- 0

        if ( abs(angle) < reward.limit )
          reward <<- cos(angle)
        else if ( i > 1 && abs(angle) > reward.limit && at.goal.previous.step )
          reward <<- -1
        else
          reward <<- 0

        at.goal.previous.step <<- abs(angle) <= reward.limit
        
        #if ( abs(angle) < reward.limit )
        #  reward <<- 0
        #else
        #  reward <<- -1

        reward <<- cos(angle) - 0.7 # This seems to be common in Japanese simulations...
        #reward <<- cos(angle) # Standard Japanese 
        #if ( abs(speed) >= 1.5 ) # This is also common in some form. Avoids spinning. 
          #reward <<- -1.7
        #else
        #if ( abs(angle) < reward.limit )
        if ( abs(angle) < pi/5 )
          reward <<- reward - abs(speed)
        
        # Decide on control action every N time steps
        #state <- c(angle, -4)
        #if ( !is.null(classif) )
          #state <- classif$get.vector(state)
        if ( (i-1)%%control.step.interval == 0 ) {
          a <<- c$set.state.vector(state, reward)
          #print(a)
          if ( is.na(a) || is.nan(a) ) {
            p$set.torque(0)
          }
          else {
            if ( a == 1 )
              p$set.torque(5)
            else if ( a == 2 )
              p$set.torque(-5)
            else
              p$set.torque(0)
          }
        }
      }
      p$step()
      if ( !is.null(v) ) 
        v$update()
    }


    if ( !is.null(v) ) {
      #v$show.action.values()
      v$show.3d.surfaces()
    }
    
    return(goal.steps*pendulum$get.time.step())
  }

  # Run "episodes" number of episodes, of length "time" seconds each.
  # Return vector of "uptime" (at.goal) in seconds for each episode. 
  run.episodes <- function(episodes, time) {
    cnt <- vector("integer", length=episodes)
    for ( ep in 1:episodes ) {
      if ( !is.null(v) ) 
        v$set.episode(ep)
      up.time <- run.time(time)
      p$reset()
      c$reset()
      cnt[ep] <- up.time
    }
    if ( !is.null(v) ) 
      v$set.episode(-1)
    return(cnt)
  }

  set.classifier <- function(classifier) {
    classif <<- classifier 
    if ( !is.null(classif) )
      nbr.classes <- classif$get.total.classes()
    else
      nbr.classes <- 2
    state <<- vector(mode = "integer", length=nbr.classes)
  }
  
  set.classifier(classifier)

  list (
        get.pendulum = function() { p },
        get.controller = function() { c },
        get.classifier = function() { classif },
        set.pendulum = function(pendulum) { p <<- pendulum },
        set.classifier = function(classifier) { set.classifier(classifier) },
        set.controller = function(controller) { c <<- controller },
        set.visualiser = function(visualiser) { v <<- visualiser },
        run.time = function(time) { run.time(time) },
        run.episodes = function(episodes, time) {
          run.episodes(episodes, time)
        },
        get.reward = function() { reward }
        )
}

# Run given number of episodes for given time using the handcoded controller.
# Default is one episode for 60 seconds (simulated time).
# Return vector containing "up time" in seconds for each episode. 
plt.run.handcoded <- function(episodes=1, simulation.time=60) {
  pendulum <- pendulum.limited.torque.new()
  control <- handcoded.pendulum.controller.new(pendulum)
  visualiser=pendulum.visualizer.new(pendulum)
  r <- plt.runner.new(pendulum, control, visualiser=visualiser)
  visualiser$set.runner(r)
  up.time <- r$run.episodes(episodes, simulation.time)
  return(up.time)
}

plt.run.sarsa <- function(nagents, nepisodes, simulation.time=60, classifier=NULL, lr=0.1, dr=0.9, lambda=0.9, epsilon=0.1, accumulating.trace = FALSE, trace.reset=FALSE, visualize=FALSE) {

  # Create pendulum object
  pendulum <- pendulum.limited.torque.new()
  #pendulum$set.angle(2.0*pi/5.0) # This is just for getting some constant plots, should normally be commented away!
  if ( visualize ) 
    visualiser <- pendulum.visualizer.new(pendulum)
  else
    visualiser <- NULL

  # Set up number of states (state variable, actually) and actions
  if ( !is.null(classifier) )
    nstates <- classifier$get.total.classes()
  else
    nstates <- 2 # If we use angle, angular speed directly
  nactions <- 2 # 2 means clockwise/counter-cw, 3 means "no action" as third

  r <- plt.runner.new(pendulum,classifier=classifier,visualiser=visualiser)
  if ( !is.null(visualiser) )
    visualiser$set.runner(r)
  cnts <- matrix(nrow=nagents, ncol=nepisodes)
  for ( agent in 1:nagents ) {
    if ( !is.null(visualiser) )
      visualiser$set.run(agent)
     # Set up SARSA learner
    sl <- sarsa.learner.new(nstates, nactions, use.trace=T)
    sl$para.setup(lr, dr, lambda)
    sl$get.policy()$set.epsilon(epsilon)
    sl$get.trace()$set.accumulating.trace(accumulating.trace);
    sl$get.trace()$set.reset.unused.actions(trace.reset);
    r$set.controller(sl)
    # Run desired number of episodes
    cnts[agent,] <- r$run.episodes(nepisodes, simulation.time)
    #print(sl$get.estimator()$get.weights())
  }
  if ( !is.null(visualiser) )
    visualiser$set.run(-1)
  return(cnts)
}

run.plt.agents <- function(nagents, neps, classifier, alphas, lambdas, accumulating.trace=FALSE, trace.reset=TRUE, var.prefix="",dr=1.0,epsilon=0.0,step.reward=-1,goal.reward=0,simulation.time=60,visualize=FALSE) {
  resvars <- get.resvar.names(var.prefix, alphas, lambdas)
  for ( i in 1:length(alphas) ) {
    res <- plt.run.sarsa(nagents,neps,simulation.time=simulation.time,classifier=classifier,lr=alphas[i],dr=dr,lambda=lambdas[i],epsilon=epsilon,accumulating.trace=accumulating.trace,trace.reset=trace.reset,visualize=visualize)
    assign(resvars[i], res, pos=.GlobalEnv)
    print(paste("Finished calculating", resvars[i]))
  }
  return(resvars)
}

run.plt <- function(plot=TRUE, save=FALSE, run.sarsa=TRUE, run.oiv=FALSE, run.bimm=FALSE) {
  nbr.agents <- 10
  nbr.episodes <- 200
  #classifier <- pendulumclassifier.new(angle.ints=10, speed.ints=10)
  #lr <- 0.9
  #lambda <- 0.99

  #classifier <- pendulumclassifier.new(cmac=TRUE, angle.ints=10, speed.ints=10, nbr.layers=5)
  #lr <- 0.2
  #lambda <- 0
  
  classifier <- pendulumclassifier.new(rbf=TRUE, angle.ints=10, speed.ints=10)
  lr <- 1
  lambda <- 0.9

  # Sarsa
  if ( run.sarsa ) {
    cnt.sarsa <- plt.run.sarsa(nbr.agents,nbr.episodes,simulation.time=60,classifier=classifier,lr=lr,dr=1,lambda=lambda,epsilon=0.1,accumulating.trace=FALSE,trace.reset=FALSE,visualize=TRUE)
  }
  
  # Sarsa, OIV
  if ( run.oiv ) {
    cnt.oiv <- sarsa.mc(nbr.agents,nbr.episodes,mcd,lr=lr,dr=1.0,lambda=0.9,epsilon=0.0,step.reward=-1,goal.reward=0)
    if ( save ) 
      write.table(cnt.oiv,"MCres/OIV_MC8x8_lr01_dr1_lmda09_e0.txt", row.names=F,col.names=F)
    if (plot)
      lines(1:nbr.episodes, colMeans(cnt.oiv), type='l', col='green')
  }
  
  # BIMM
  if ( run.bimm ) {
    cnt.bimm <- bimm.mc(nbr.agents,nbr.episodes,mcd, k.stm=0.000001,k.ltm=1.0,slap.lr=1,sarsa.lr=lr,dr=0.9,lambda=0.95,epsilon=0.0,step.reward=0.0,goal.reward=1.0)
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

