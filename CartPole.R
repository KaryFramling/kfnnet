# Cart-Pole swing up task with limited torque formulas taken
# from Schaal, 1997 but should be identical with both older
# and newer sources.
#
# Created 14.8.2006, Kary Främling
cartpole.new <- function() {

  g <- 9.81 # Gravity. Negative sign in Barto et al (1983) is error (tested!).
  m.cart <- 1 # 1 kg
  m.pole <- 0.1 # kgs
  l <- 0.5 # meter, half-pole length
  mu.cart <- 0.0005 # Friction of cart on track
  mu.pole <- 0.000002 # Friction of pole on cart
  force.max <- 10.0 # Max/min (+/-) force applied to cart
  cart.width <- 1.2
  cart.height <- 0.5
  track.half.length <- 3 # 3 meters to both sides
  wheel.radius <- 0.2

  up.angle.limit <- pi/15 # Tilt angle over 12 degrees (12/180) is failure

  time <- 0.0 # Current time
  time.step <- 0.02 # Seconds. Same as in Barto et al., 1983.
  
  init.theta <- 0.0
  theta <- init.theta # Pendulum angle, radians. 
  dtheta <- 0.0 # Pendulum angular speed, radians/s. Initial value zero.
  ddtheta <- 0.0 # Angular acceleration, radians/s^2.
  x <- 0 # Cart position
  dx <- 0 # Cart velocity
  ddx <- 0 # Cart acceleration
  force <- 0.0 # Applied force

  reset <- function() {
    theta <<- init.theta # As in Barto et al., 1983
    #theta <<- -up.angle.limit + runif(1)*2*up.angle.limit # More difficult!
    dtheta <<- 0.0
    ddtheta <<- 0.0
    x <<- 0
    dx <<- 0
    ddx <<- 0
    force <<- 0.0
    time <<- 0.0
  }

  next.theta <- function(theta, dtheta) {
    res <- theta + dtheta*time.step;
    return(res)
  }

  next.dtheta <- function(dtheta, ddtheta) {
    res <- dtheta + ddtheta*time.step
    return(res)
  }

  # Get new acceleration after one time step. 
  next.ddtheta <- function(force, theta, dtheta, x, dx) {
    st <- sin(theta)
    ct <- cos(theta)
    t1 <- g*st
    t2 <-ct*((-force - m.cart*l*dtheta*dtheta*st + mu.cart*sign(dx))/(m.cart + m.pole))
    t3 <- mu.pole*dtheta/m.pole*l
    res <- (t1 + t2 + t3)/(l*(4/3 - m.pole*ct*ct/(m.cart + m.pole)))
    return(res)
  }

  next.x <- function(x, dx) {
    res <- x + dx*time.step;
    return(res)
  }

  next.dx <- function(dx, ddx) {
    res <- dx + ddx*time.step
    return(res)
  }

  # Get new acceleration after one time step. 
  next.ddx <- function(force, theta, dtheta, ddtheta, x, dx) {
    st <- sin(theta)
    ct <- cos(theta)
    res <- (force + m.pole*l*(dtheta*dtheta*st - ddtheta*ct) - mu.cart*sign(dx))/(m.cart + m.pole)
    return(res)
  }

  step = function() {
    time <<- time + time.step
    ddtheta <<- next.ddtheta(force, theta, dtheta, x, dx)
    dtheta <<- next.dtheta(dtheta, ddtheta)
    theta <<- next.theta(theta, dtheta)
    ddx <<- next.ddx(force, theta, dtheta, ddtheta, x, dx)
    dx <<- next.dx(dx, ddx)
    x <<- next.x(x, dx)
  }

  has.failed <- function() {
    if ( abs(theta) > up.angle.limit )
      return (TRUE)
    if ( x-cart.width/2 <= -track.half.length || x+cart.width/2 >= track.half.length )
      return (TRUE)
    return(FALSE)
  }

  get.action.descriptions <- function() {
    c("Right", "Left")
  }
  
  reset() # Initialize

  # Return list of "public" functions
  list (
        get.track.half.length = function() { track.half.length },
        get.cart.width = function() { cart.width },
        get.cart.height = function() { cart.height },
        get.wheel.radius = function() { wheel.radius },
        reset = function() { reset() },
        get.time = function() { time },
        get.time.step = function() { time.step },
        get.angle = function() { theta },
        get.angle.speed = function() { dtheta },
        get.angle.acc = function() { ddtheta },
        get.x = function() { x },
        get.dx = function() { dx },
        get.ddx = function() { ddx },
        get.force = function() { force },
        get.init.theta = function() { init.theta },
        get.up.angle.limit = function() { up.angle.limit },
        set.force = function(value) { force <<- value },
        set.time.step = function(value) { time.step <<- value },
        set.init.theta = function(value) { init.theta <<- value },
        set.up.angle.limit = function(value) { up.angle.limit <<- value },
        has.failed = function() { has.failed() },
        step = function() { step() },
        get.action.descriptions = function() { get.action.descriptions() }
        )
}

# Pendulum input space discretizer. Requires "Functions.R" to be run first!
cartpole.discretizer.new <- function(rbf=FALSE, cmac=FALSE,x.min=-2.4, x.max=2.4, x.ints=3, theta.min=-pi/15, theta.max=pi/15, theta.ints=6, dx.min=-1.5, dx.max=1.5, dx.ints=3, dtheta.min=-5*pi/6, dtheta.max=5*pi/6, dtheta.ints=3, nbr.layers=5) {

  dims <- 4
  mins <- c(x.min,theta.min,dx.min,dtheta.min)
  maxs <- c(x.max,theta.max,dx.max,dtheta.max)
  ncls <- c(x.ints,theta.ints,dx.ints,dtheta.ints)
  if ( cmac ) {
    discretizer <- cmac.new(nbr.layers=nbr.layers, nbr.dimensions=dims, minvals=mins, maxvals=maxs, nbr.classes=ncls, random.offset=TRUE)
  }
  else if ( rbf ) {
    rbfc <- rbf.classifier.new(nbrInputs=dims, nbrOutputs=0, 
                               activation.function=squared.distance.activation,
                               output.function=gaussian.output.function #imqe.output.function
                               )
    at <- scale.translate.ranges(mins, maxs, c(0,0,0,0), c(1,1,1,1))
    rbfc$init.centroids.grid(mins, maxs, ncls, affine.transformation=at)
    #rbfc$set.spread(1/max(ncls))
    rbfc$set.spread(0.05)
    rbfc$set.normalize(TRUE)
    discretizer <- rbfc
  }
  else {
    # Ordinary lookup-table
    discretizer <- discrete.classifier.new(nbr.dimensions=dims, minvals=mins, maxvals=maxs, nbr.classes=ncls)
    # Uneven intervals for theta must be treated separately.
    discretizer$set.discretizer(2, uneven.discretizer.new(c(-pi/30,-pi/180,0,pi/180,pi/30)))
  }

  list (
        get.total.classes = function() { discretizer$get.total.classes() },
        get.vector = function(state) { discretizer$get.vector(state) },
        get.class = function(x, angle, dx, dangle) {
          return(discretizer$get.index(c(x, angle, dx, dangle)))
        }
        )
}

test.cartpole.discretizer <- function() {
  d <- cartpole.discretizer.new()
  for ( da in c(-100,0,100) )
    for ( dx in -1:1 )
      for ( a in c(-8,-4,-0.5,0.5,4,8) )
        for ( x in -1:1 )
          print(d$get.class(x,a,dx,da))
}

# Pendulum visualizer
cartpole.visualizer.new <- function(cartpole, r=NULL, nbr.plots=1) {
  cp <- cartpole
  runner <- r
  action.value.win <- 0
  
  update <- function() {
    # Number of plots we want to include
    if ( nbr.plots == 2 )
      par(mfcol=c(1,2))
    else if ( nbr.plots == 3 || nbr.plots == 4 )
      par(mfcol=c(2,2))

    # Create window if we don't have one yet
    hl <- cp$get.track.half.length()
    xpoints <- c(-hl - 0.5, hl + 0.5)
    ypoints <- xpoints
    plot(xpoints, ypoints, type = "n", xlab = "", ylab = "", axes = F,
         main = "Cart-Pole balancing")
    cat("")

    # Draw track
    lines(x=c(xpoints[1],-hl,-hl,hl,hl,xpoints[2]),y=c(0.3,0.3,0,0,0.3,0.3),col="black")
    
    # Draw new position
    x <- cp$get.x()
    w <- cp$get.cart.width()
    half.w <- w/2
    h <- cp$get.cart.height()
    xp <- c(x-half.w, x+half.w, x+half.w, x-half.w, x-half.w)
    yp <- c(0, 0, h, h, 0)
    lines(x=xp,y=yp,col="black") # Cart
    
    angle <- cp$get.angle()
    lines(x=c(x, x+sin(angle)),y=c(h, h+cos(angle)),col="black") # Pole
    points(x=x+sin(angle),y=h+cos(angle),col="black",pch=19) # Weight at end
    cat("")

    # Labels indicating different values
    text(xpoints[1],ypoints[2],labels=paste("Time:", cp$get.time()),pos=4)
    text(xpoints[1],ypoints[2]-0.3,labels=paste("X:", cp$get.x()),pos=4)
    text(xpoints[1],ypoints[2]-0.6,labels=paste("Angle:", angle),pos=4)
    text(xpoints[1],ypoints[2]-0.9,labels=paste("dX:", cp$get.dx()),pos=4)
    text(xpoints[1],ypoints[2]-1.2,labels=paste("dAngle:", cp$get.angle.speed()),pos=4)
    text(xpoints[1],ypoints[2]-1.5,labels=paste("Force:", cp$get.force()),pos=4)
    if ( !is.null(runner) ) {
      text(xpoints[1],ypoints[2]-1.8,labels=paste("Reward:", runner$get.reward()),pos=4)
      text(xpoints[1],ypoints[2]-2.1,labels=paste("Episode:", runner$get.episode()),pos=4)
    }
  }

  show.action.values <- function(phi=35) {
    # Take care of having our own window. 
    old.dev <- dev.cur()
    if ( action.value.win == 0 ) {
      x11()
      par(mfcol=c(2,2))
      action.value.win <<- dev.cur()
    }
    dev.set(action.value.win)
    
    # Then plot
    fa <- runner$get.controller()$get.estimator()
    classif <- runner$get.classifier()
    cfa <- classifying.function.approximator.new(fa, classif)
    inps <- vector(mode="numeric", length=4)
    plot3d.output(cfa, show.indices=c(1,2), show.mins=c(-2.4,-pi/12+0.1), show.maxs=c(2.4,pi/12), show.steps=c(0.2,pi/90), outindex=1, default.inputs=inps, theta=135, phi=phi, shade=0.3, xlab="x", ylab="pole angle")
    plot3d.output(cfa, show.indices=c(2,3), show.mins=c(-pi/12+0.1,-1.5), show.maxs=c(pi/12,1.5), show.steps=c(pi/90,0.2), outindex=1, default.inputs=inps, theta=135, phi=phi, shade=0.3, xlab="pole angle", ylab="cart speed")
    plot3d.output(cfa, show.indices=c(1,2), show.mins=c(-2.4,-pi/12+0.1), show.maxs=c(2.4,pi/12), show.steps=c(0.2,pi/90), outindex=2, default.inputs=inps, theta=135, phi=phi, shade=0.3, xlab="x", ylab="pole angle")
    plot3d.output(cfa, show.indices=c(2,3), show.mins=c(-pi/12+0.1,-1.5), show.maxs=c(pi/12,1.5), show.steps=c(pi/90,0.2), outindex=2, default.inputs=inps, theta=135, phi=phi, shade=0.3, xlab="pole angle", ylab="cart speed")
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
    show.mins=c(-2.4,-pi/12)
    show.maxs=c(2.4,pi/12)
    show.steps=c(0.2,pi/90)
    vals <- list()
    vals[[1]] <- seq(show.mins[1], show.maxs[1], show.steps[1])
    vals[[2]] <- seq(show.mins[2], show.maxs[2], show.steps[2])

    # Then plot
    inps <- vector(mode="numeric", length=4)
    m <- create.input.matrix(indices=show.indices, mins=show.mins, maxs=show.maxs, steps=show.steps, default.inputs=inps)
    outs <- apply(m, 1, cfa$eval)
    z <- outs[1,]
    zm <- matrix(data=z, nrow=length(vals[[1]]), ncol=length(vals[[2]]), byrow=TRUE)
    zlim <- get.z.range(zm)
    persp(vals[[1]], vals[[2]], zm, zlim=zlim, theta=135, phi=35, shade=0.3, xlab="x", ylab="pole angle", main=cp$get.action.descriptions()[1], ticktype="detailed", ...)
    z <- outs[2,]
    zlim <- get.z.range(z)
    zm <- matrix(data=z, nrow=length(vals[[1]]), ncol=length(vals[[2]]), byrow=TRUE)
    persp(vals[[1]], vals[[2]], zm, zlim=zlim, theta=135, phi=35, shade=0.3, xlab="x", ylab="pole angle", main=cp$get.action.descriptions()[2], ticktype="detailed", ...)
    z <- apply(outs, 2, max)
    zlim <- get.z.range(z)
    zm <- matrix(data=z, nrow=length(vals[[1]]), ncol=length(vals[[2]]), byrow=TRUE)
    persp(vals[[1]], vals[[2]], zm, zlim=zlim, theta=135, phi=35, shade=0.3, xlab="x", ylab="pole angle", main="Action-value function (by MAX)", ticktype="detailed", ...)
    z <- apply(outs, 2, diff)
    zlim <- get.z.range(z)
    zm <- matrix(data=z, nrow=length(vals[[1]]), ncol=length(vals[[2]]), byrow=TRUE)
    persp(vals[[1]], vals[[2]], zm, zlim=zlim, theta=135, phi=35, shade=0.3, xlab="x", ylab="pole angle", main="Action-value difference between actions", ticktype="detailed", ...)
    #contour(vals[[1]], vals[[2]], zm, xlab="x", ylab="pole angle")
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
        update = function() { update() },
        show.action.values = function() { show.action.values() },
        show.3d.surfaces = function(...) { show.3d.surfaces(...) }
        )
}

# Random-action controller. Implements "Controller" interface. 
random.cartpole.controller.new <- function(cartpole=NULL) {

  cp <- cartpole
  state <- NULL
  action <- NULL
  
  set.state.vector <- function(inputs, reward) {
    state <<- inputs
    return(c(round(1 + runif(1))))
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
              get.action.descriptions = function() { cp$get.action.descriptions() },
              get.task = function() { cp },
              set.task = function(task) { cp <<- task }
              )

  # We implement "Controller" interface
  c <- controller.new()

  class(pub) <- c("RandomCartPoleController",class(c))
  return(pub)
}

# Handcoded controller for pendulum task. Implements "Controller" interface. 
handcoded.cartpole.controller.new <- function(cartpole=NULL) {

  cp <- cartpole
  state <- NULL
  action <- NULL
  
  set.state.vector <- function(inputs, reward) {
    state <<- inputs
    angle <- state[2]
    if ( angle > 0 )
      action <<- 1
    else
      action <<- 2
    return(action)
  }

  get.action.descriptions <- function() {
    c("Right", "Left")
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
              get.action.descriptions = function() { get.action.descriptions() },
              get.task = function() { cp },
              set.task = function(task) { cp <<- task }
              )

  # We implement "Controller" interface
  c <- controller.new()

  class(pub) <- c("HandCodedCartPoleController",class(c))
  return(pub)
}

# Handcoded controller for Swing-Up pendulum task.
# Implements "Controller" interface. 
handcoded.swingupcartpole.controller.new <- function(cartpole=NULL) {

  cp <- cartpole
  state <- NULL
  action <- NULL
  
  set.state.vector <- function(inputs, reward) {
    state <<- inputs
    x <- state[1]
    angle <- state[2]
    dx <- state[3]

    # Limit on speed
    if ( dx > 2 )
      return(2)
    else if ( dx < -2 )
      return(1)
    
    # Limit on position
    if ( x > 2 && dx > 0 )
      return(2)
    else if ( x < -2 && dx < 0 )
      return(1)

    # Try to stop at upright position
    #action <<- min(1,floor(runif(1)))
    if ( abs(angle) < pi/15 ) {
      if ( angle > 0 )
        action <<- 1
      else
        action <<- 2
    }
    else {
      if ( angle > 0 )
        action <<- 2
      else
        action <<- 1
    }
    return(action)
  }

  get.action.descriptions <- function() {
    c("Right", "Left")
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
              get.action.descriptions = function() { get.action.descriptions() },
              get.task = function() { cp },
              set.task = function(task) { cp <<- task }
              )

  # We implement "Controller" interface
  c <- controller.new()

  class(pub) <- c("HandCodedSwingupCartPoleController",class(c))
  return(pub)
}

# "Class" for running Cart-Pole task.
cpt.runner.new <- function(cartpole, controller=NULL, classifier=NULL, visualiser=NULL) {
  cp <- cartpole
  c <- controller
  classif <- classifier
  v <- visualiser
  state <- vector(mode = "integer", length=2) # Default direct values
  reward <- 0
  episode <- 0

  get.real.time <- function(steps) {
    steps*cp$get.time.step()
  }
  
  get.state.vector <- function() {
    x <-cp$get.x()
    angle <-cp$get.angle()
    dx <-cp$get.dx()
    angle.speed <-cp$get.angle.speed()
    state <<- c(x, angle, dx, angle.speed)
    return (state)
  }

  # Runs until failure or until max.time is exceeded.
  # Returns time before failure in seconds. 
  run.until.failure <- function(max.time=120) {
    steps <- 0
    while ( !cp$has.failed() && get.real.time(steps) <= max.time ) {
      # Update state
      state <<- get.state.vector()
      angle <- state[2]
      if ( !is.null(classif) )
        state <<- classif$get.vector(state)
      #print(state)

      if ( !is.null(c) ) {
        # Give reward. This is constantly zero until failure.
        reward <<- 0
        #reward <<- cos(angle) # Just testing

        # Get next action
        a <- c$set.state.vector(state, reward) 
        if ( a == 1 )
          cp$set.force(10)
        else if ( a == 2 )
          cp$set.force(-10)
      }
      cp$step()
      steps <- steps + 1
      if ( !is.null(v) ) 
        v$update()
    }

    # Last negative reward after failure
    if ( cp$has.failed() ) {
      state <<- get.state.vector()
      if ( !is.null(classif) )
        state <<- classif$get.vector(state)
      reward <<- -1
      a <- c$set.state.vector(state, reward)
    }

    if ( !is.null(v) ) {
      #v$show.action.values()
      v$show.3d.surfaces()
    }
    
    return(get.real.time(steps))
  }

  # Run "episodes" number of episodes.
  # Return vector of "uptime" (no failure) in seconds for each episode. 
  run.episodes <- function(episodes, max.time) {
    cnt <- vector("integer", length=episodes)
    for ( ep in 1:episodes ) {
      episode <<- ep
      up.time <- run.until.failure(max.time)
      #print(c$get.estimator()$get.weights())
      cp$reset()
      c$reset()
      cnt[episode] <- up.time
    }
    return(cnt)
  }

  set.classifier <- function(classifier) {
    classif <<- classifier 
    if ( !is.null(classif) )
      nbr.classes <- classif$get.total.classes()
    else
      nbr.classes <- 4
    state <<- vector(mode = "integer", length=nbr.classes)
  }
  
  #set.classifier(NULL)

  list (
        get.episode = function() { episode },
        get.cartpole = function() { cp },
        get.controller = function() { c },
        get.classifier = function() { classif },
        set.cartpole = function(cartpole) { cp <<- cartpole },
        set.classifier = function(classifier) { set.classifier(classifier) },
        set.controller = function(controller) { c <<- controller },
        set.visualiser = function(visualiser) { v <<- visualiser },
        run.until.failure = function(max.time) { run.until.failure(max.time) },
        run.episodes = function(episodes, max.time) {
          run.episodes(episodes, max.time)
        },
        get.reward = function() { reward }
        )
}


# Run given number of episodes for given time using passed controller.
# Default is one episode for 60 seconds (simulated time).
# Return vector containing "up time" in seconds for each episode. 
cpt.run <- function(controller, episodes=1, max.time=60, cartpole=NULL) {
  if ( is.null(cartpole) )
    cartpole <- cartpole.new()
  controller$set.task(cartpole)
  visualiser <- cartpole.visualizer.new(cartpole)
  r <- cpt.runner.new(cartpole, controller, visualiser=visualiser)
  visualiser$set.runner(r)
  up.time <- r$run.episodes(episodes, max.time)
  return(up.time)
}


# Run given number of episodes for given time using the "random" controller.
# Default is one episode for 60 seconds (simulated time).
# Return vector containing "up time" in seconds for each episode. 
cpt.run.random <- function(episodes=1, max.time=60) {
  control <- random.cartpole.controller.new()
  up.time <- cpt.run(control, episodes, max.time)
  return(up.time)
}

# Run given number of episodes for given time using the handcoded controller.
# Default is one episode for 60 seconds (simulated time).
# Return vector containing "up time" in seconds for each episode. 
cpt.run.handcoded <- function(episodes=1, max.time=120) {
  control <- handcoded.cartpole.controller.new()
  up.time <- cpt.run(control, episodes, max.time)
  return(up.time)
}

# Run with swing-up functionality for given number of episodes and time
# using handcoded controller.
# Default is one episode for 60 seconds (simulated time).
# Return vector containing "up time" in seconds for each episode. 
cpt.run.handcoded.swingup <- function(episodes=1, max.time=120) {
  cp <- cartpole.new()
  cp$set.up.angle.limit(Inf) # No limit
  cp$set.init.theta(pi) # Always start "head down". Maybe random angle would be better?
  cp$reset()
  control <- handcoded.swingupcartpole.controller.new()
  up.time <- cpt.run(control, episodes, max.time, cp)
  return(up.time)
}

cpt.run.sarsa <- function(nagents, nepisodes, max.time=120, classifier=NULL, lr=0.1, dr=0.9, lambda=0.9, epsilon=0.1, accumulating.trace = FALSE, trace.reset=FALSE, use.softmax=FALSE, visualize=FALSE) {

  # Create pendulum object
  cartpole <- cartpole.new()
  if ( visualize ) 
    visualiser <- cartpole.visualizer.new(cartpole, nbr.plots=1)
  else
    visualiser <- NULL

  # Set up number of states (state variable, actually) and actions
  if ( !is.null(classifier) )
    nstates <- classifier$get.total.classes()
  else
    nstates <- 4 # If we use control parameters
  nactions <- 2

  # Go for it
  r <- cpt.runner.new(cartpole,classifier=classifier,visualiser=visualiser)
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
    # Use Boltzmann (Softmax) instead of e-greedy?
    if ( use.softmax ) {
      bp <- boltzmann.new(temperature=0.0001) 
      sl$set.policy(bp)
    }
    r$set.controller(sl)
    # Run desired number of episodes
    cnts[agent,] <- r$run.episodes(nepisodes, max.time)
    #print(sl$get.estimator()$get.weights())
  }
  return(cnts)
}

run.cpt.agents <- function(nagents, neps, classifier, alphas, lambdas, accumulating.trace=FALSE, trace.reset=TRUE, var.prefix="",dr=1.0,epsilon=0.1,max.time=240,use.softmax=FALSE,visualize=FALSE) {
  resvars <- get.resvar.names(var.prefix, alphas, lambdas)
  for ( i in 1:length(alphas) ) {
    res <- cpt.run.sarsa(nagents,neps,max.time=max.time,classifier=classifier,lr=alphas[i],dr=dr,lambda=lambdas[i],epsilon=epsilon,accumulating.trace=accumulating.trace,trace.reset=trace.reset,use.softmax=use.softmax,visualize=visualize)
    assign(resvars[i], res, pos=.GlobalEnv)
    print(paste("Finished calculating", resvars[i]))
  }
  return(resvars)
}

run.cpt <- function(plot=TRUE, save=FALSE, run.sarsa=TRUE, run.oiv=FALSE, run.bimm=FALSE) {
  nbr.agents <- 2
  nbr.episodes <- 200
  max.time <- 240
  #classifier <- cartpole.discretizer.new()
  #lr <- 0.05
  #lambda <- 0.5
  #classifier <- cartpole.discretizer.new(cmac=T, nbr.layers=10)
  #lr <- 0.01
  #lambda <- 0.5
  classifier <- cartpole.discretizer.new(rbf=TRUE, x.ints=6, theta.ints=6, dx.ints=6, dtheta.ints=6)
  lr <- 3
  lambda <- 0

  # Sarsa
  if ( run.sarsa ) {
    cnt.sarsa <- cpt.run.sarsa(nbr.agents,nbr.episodes,max.time=max.time,classifier=classifier,lr=lr,dr=0.98,lambda=lambda,epsilon=0.1,accumulating.trace=FALSE,trace.reset=FALSE,use.softmax=F,visualize=TRUE)
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

