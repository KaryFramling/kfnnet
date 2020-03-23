# "R" routines for performing tests on the use of NLMS as normalisation
# in NRBF networks (and whatever extensions and other tests that leads to).
#
# Kary Främling, created 1 jan 2008
#

# Everything needed for this file to work
source("Interfaces.R")
source("Functions.R")
source("NeuralLayer.R")
source("Adaline.R")
source("RBF.R")

# Do training for a whole training set, one sample at a time.
# Training is here done sequentially, in most "real-world"
# applications samples would rather be randomly selected. 
train.whole.set <- function(inputs, targets, net, iterations=1) {
  for ( iter in 1:iterations ) {
    for ( row in 1:nrow(inputs) ) {
      out <- net$eval(inputs[row,])
      net$train(targets[row,])
    }
  }
}

plot.rbf.vs.nrbf.example <- function() {
    rbf <- rbf.new(1, 1, 0, activation.function=squared.distance.activation, 
                 output.function=imqe.output.function)
    ol <- rbf$get.outlayer()
    ol$set.use.bias(TRUE)
    ol$set.weights(matrix(c(-2,2,3),nrow=1))
    hl <- rbf$get.hidden()
    hl$set.weights(matrix(c(-5,5),nrow=2,ncol=1,byrow=T))
    rbf$set.spread(1)
    x <- matrix(seq(-15,15,0.1))
    y <- rbf$eval(x)
    plot(x, y, type='l')
    ol$set.use.bias(FALSE)
    ol$set.weights(matrix(c(1,5),nrow=1))
    rbf$set.nrbf(TRUE)
    y <- rbf$eval(x)
    lines(x, y)
}

nlms.vs.nrbf.2D <- function() {
  #rbf <- rbf.new(1, 1, 0, activation.function=squared.distance.activation, 
  #               output.function=gaussian.output.function)
  
  targets <- matrix(c(1,5),nrow=2,ncol=1)
  tx <- matrix(c(1,5),nrow=2,ncol=1)
  #targets <- matrix(c(5,1,5),nrow=3,ncol=1)
  #tx <- matrix(c(-1,1,5),nrow=3,ncol=1)
  #targets <- matrix(c(5,5),nrow=2,ncol=1)
  #tx <- matrix(c(-2,2),nrow=2,ncol=1)
  rbf <- rbf.new(1, 1, 0, activation.function=squared.distance.activation, 
                 output.function=imqe.output.function)
  #rbf <- rbf.new(1, 1, 0, activation.function=squared.distance.activation, 
  #               output.function=gaussian.output.function)
  ol <- rbf$get.outlayer()
  ol$set.use.bias(TRUE)
  ol$set.weights(matrix(c(0,0,0),nrow=1))
  hl <- rbf$get.hidden()
  hl$set.weights(matrix(c(-5,5),nrow=2,ncol=1,byrow=T))
  rbf$set.spread(0.1)
  xlo <- -10
  xhi <- 10
  ylo <- -5
  yhi <- 6
  x <- matrix(seq(xlo,xhi,0.1))
  lwd <- 2
  lty <- 1
  plot(c(xlo,xhi),c(ylo,yhi),type='n')
  legend("topleft", c("RBF", "RBF/NLMS", "NRBF", "NRBF/NLMS"), lty=c(1,2,3,4), lwd=2)
  # Set up experiments
  nbr.itrs <- 10
  lrs <- c(0.1,0.5,1.0,1.5,2.0,3.0)
  rbfres <- matrix(0,nrow=length(lrs),ncol=nbr.itrs)
  rbfnlmsres <- matrix(0,nrow=length(lrs),ncol=nbr.itrs)
  nrbfres <- matrix(0,nrow=length(lrs),ncol=nbr.itrs)
  nrbfnlmsres <- matrix(0,nrow=length(lrs),ncol=nbr.itrs)
  for ( lrindex in 1:length(lrs) ) {
    ol$set.lr(lrs[lrindex])
    # RBF, no normalisation
    rbf$set.nrbf(FALSE)
    ol$set.use.bias(TRUE)
    ol$set.nlms(FALSE)
    ol$set.weights(matrix(c(0,0,0),nrow=1))
    for ( iteration in 1:nbr.itrs ) {
      train.whole.set(tx, targets, rbf, iterations=1)
    #print(ol$get.weights())
      ty <- rbf$eval(tx)
      rbfres[lrindex,iteration] <- root.mean.squared.error(targets,ty)
    #print(ty)
    y <- rbf$eval(x)
    lines(x, y, lwd=lwd, lty=lty)
    }
    # RBF with NLMS
    ol$set.nlms(TRUE)
    ol$set.weights(matrix(c(0,0,0),nrow=1))
    for ( iteration in 1:nbr.itrs ) {
      train.whole.set(tx, targets, rbf, iterations=1)
      #print(ol$get.weights())
      ty <- rbf$eval(tx)
      rbfnlmsres[lrindex,iteration] <- root.mean.squared.error(targets,ty)
      #print(ty)
      y <- rbf$eval(x)
      #lty <- lty+1
      lines(x, y, lwd=lwd, lty=lty)
    }
    #  "Ordinary" NRBF
    ol$set.use.bias(FALSE)
    ol$set.weights(matrix(c(0,0),nrow=1))
    rbf$set.nrbf(TRUE)
    ol$set.nlms(FALSE)
    for ( iteration in 1:nbr.itrs ) {
      train.whole.set(tx, targets, rbf, iterations=1)
      ty <- rbf$eval(tx)
      nrbfres[lrindex,iteration] <- root.mean.squared.error(targets,ty)
      #print(ty)
      y <- rbf$eval(x)
      #lty <- lty+1
      lines(x, y, lwd=lwd, lty=lty)
    }
    # NRBF with NLMS
    ol$set.weights(matrix(c(0,0),nrow=1))
    ol$set.nlms(TRUE)
    for ( iteration in 1:nbr.itrs ) {
      train.whole.set(tx, targets, rbf, iterations=1)
      ty <- rbf$eval(tx)
      nrbfnlmsres[lrindex,iteration] <- root.mean.squared.error(targets,ty)
     #print(ty)
      y <- rbf$eval(x)
      #lty <- lty+1
      lines(x, y, lwd=lwd, lty=lty)
    }
  }
  print(rbfres)
  print(rbfnlmsres)
  print(nrbfres)
  print(nrbfnlmsres)
}

nlms.vs.nrbf.many.RBF <- function() {
  #rbf <- rbf.new(1, 1, 0, activation.function=squared.distance.activation, 
  #               output.function=gaussian.output.function)
  
  targets <- matrix(c(2,5),nrow=2,ncol=1)
  tx <- matrix(c(-2,5),nrow=2,ncol=1)
  #targets <- matrix(c(5,1,5),nrow=3,ncol=1)
  #tx <- matrix(c(-1,1,5),nrow=3,ncol=1)
  #targets <- matrix(c(5,5),nrow=2,ncol=1)
  #tx <- matrix(c(-2,2),nrow=2,ncol=1)
  rbf <- rbf.new(1, 1, 0, activation.function=squared.distance.activation, 
                 output.function=imqe.output.function)
  #rbf <- rbf.new(1, 1, 0, activation.function=squared.distance.activation, 
  #               output.function=gaussian.output.function)
  ol <- rbf$get.outlayer()
  hl <- rbf$get.hidden()
  rbf$set.spread(0.1)
  xlo <- -10
  xhi <- 10
  ylo <- 0
  yhi <- 6
  x <- matrix(seq(xlo,xhi,0.1))
  lwd <- 2
  lty <- 1
  plot(c(xlo,xhi),c(ylo,yhi),type='n')
  legend("topleft", c("RBF/Bias", "RBF/NLMS/Bias", "RBF", "RBF/NLMS", "NRBF", "NRBF/NLMS"), lty=c(1,2,3,4,5,6), lwd=2)
  # Set up experiments
  nbr.itrs <- 10

  # RBF, LMS, Bias
  ol$set.use.bias(TRUE)
  rbf$set.nrbf(FALSE)
  ol$set.nlms(FALSE)
  ol$set.lr(0.1)
  ol$set.weights(matrix(c(0,0,0,0,0,0,0,0,0,0),nrow=1))
  hl$set.weights(matrix(c(-5,-4,-3,-2.5,-2,-1.5,-1,-0.5,5),nrow=9,ncol=1,byrow=T))
  for ( iteration in 1:nbr.itrs ) {
    train.whole.set(tx, targets, rbf, iterations=1)
  }
  y <- rbf$eval(x)
  lines(x, y, lwd=lwd, lty=lty)
  
  # RBF with NLMS, Bias
  ol$set.nlms(TRUE)
  ol$set.lr(1.0)
  ol$set.weights(matrix(c(0,0,0,0,0,0,0,0,0,0),nrow=1))
  for ( iteration in 1:nbr.itrs ) {
    train.whole.set(tx, targets, rbf, iterations=1)
  }
  y <- rbf$eval(x)
  lty <- lty+1
  lines(x, y, lwd=lwd, lty=lty)

  x11()
  plot(c(xlo,xhi),c(ylo,yhi),type='n')
  
  # RBF with LMS, No Bias
  ol$set.use.bias(FALSE)
  ol$set.nlms(FALSE)
  ol$set.lr(0.1)
  ol$set.weights(matrix(c(0,0,0,0,0,0,0,0,0),nrow=1))
  for ( iteration in 1:nbr.itrs ) {
    train.whole.set(tx, targets, rbf, iterations=1)
  }
  y <- rbf$eval(x)
  lty <- lty+1
  lines(x, y, lwd=lwd, lty=lty)
  
  # RBF with NLMS, No Bias
  ol$set.nlms(TRUE)
  ol$set.lr(1.0)
  ol$set.weights(matrix(c(0,0,0,0,0,0,0,0,0),nrow=1))
  for ( iteration in 1:nbr.itrs ) {
    train.whole.set(tx, targets, rbf, iterations=1)
  }
  y <- rbf$eval(x)
  lty <- lty+1
  lines(x, y, lwd=lwd, lty=lty)
  
  x11()
  plot(c(xlo,xhi),c(ylo,yhi),type='n')

  #  "Ordinary" NRBF
  ol$set.use.bias(FALSE)
  rbf$set.nrbf(TRUE)
  ol$set.nlms(FALSE)
  ol$set.lr(1.0)
  ol$set.weights(matrix(c(0,0,0,0,0,0,0,0,0),nrow=1))
  for ( iteration in 1:nbr.itrs ) {
    train.whole.set(tx, targets, rbf, iterations=1)
  }
  y <- rbf$eval(x)
  lty <- lty+1
  lines(x, y, lwd=lwd, lty=lty)

  # NRBF with NLMS
  ol$set.nlms(TRUE)
  ol$set.lr(1.0)
  ol$set.weights(matrix(c(0,0,0,0,0,0,0,0,0),nrow=1))
  for ( iteration in 1:nbr.itrs ) {
    train.whole.set(tx, targets, rbf, iterations=1)
  }
  y <- rbf$eval(x)
  lty <- lty+1
  lines(x, y, lwd=lwd, lty=lty)
}

nlms.vs.nrbf.sombrero <- function() {
  rbf <- rbf.new(2, 1, 0, activation.function=squared.distance.activation, 
                 output.function=imqe.output.function)
  n.training.samples <- 500
  xmin <- -10
  xmax <- 10
  ymin <- -10
  ymax <- 10  
  tx <- xmin + runif(n.training.samples)*(xmax-xmin)
  ty <- ymin + runif(n.training.samples)*(ymax-ymin)
  t.in <- matrix(c(tx,ty), ncol=2)
  targets <- matrix(get.sombrero.3D(tx,ty), ncol=1)
  # trans<-persp(x=c(-10,10),y=c(-10,10),z=matrix(c(-0.2,1),nrow=2,ncol=2))
  # trans3d(x,y,z, pmat)
  # points(p3d)
  nbr.rbf.grid.x <- 11 # Would be more general to have one number per dimension.
  nbr.rbf.grid.y <- nbr.rbf.grid.x
  rbf.dist.x <- (xmax - xmin)/nbr.rbf.grid.x
  rbf.centers.x <- seq(xmin + rbf.dist.x/2, xmax, rbf.dist.x)
  rbf.centers.y <- rbf.centers.x
  rbf.centers <- create.permutation.matrix(list(rbf.centers.x, rbf.centers.y)) # My own function, built-in fctn probably exists also
  ol <- rbf$get.outlayer()
  ol$set.use.bias(FALSE)
  ol$set.weights(matrix(rep(0, nbr.rbf.grid.x*nbr.rbf.grid.y),nrow=1))
  hl <- rbf$get.hidden()
  hl$set.weights(rbf.centers)
  rbf$set.spread(5)

  # Set up experiments
  nbr.itrs <- 20

  # RBF, no normalisation. 
  rbf$set.nrbf(FALSE)
  ol$set.use.bias(FALSE)
  ol$set.nlms(FALSE)
  smb.rbf.lrs <- c(0.01,0.05,0.1,0.5,1.0)
  smb.rbf.res <- matrix(0,nrow=length(smb.rbf.lrs),ncol=nbr.itrs)
  for ( lrindex in 1:length(smb.rbf.lrs) ) {
    ol$set.lr(smb.rbf.lrs[lrindex])
    ol$set.weights(matrix(rep(0, nbr.rbf.grid.x*nbr.rbf.grid.y),nrow=1))
    for ( iteration in 1:nbr.itrs ) {
      train.whole.set(t.in, targets, rbf, iterations=1)
      ty <- rbf$eval(t.in)
      smb.rbf.res[lrindex,iteration] <- root.mean.squared.error(targets,ty)
    }
  }
  
  # RBF, NLMS. 
  rbf$set.nrbf(FALSE)
  ol$set.use.bias(FALSE)
  ol$set.nlms(TRUE)
  smb.rbf.nlms.lrs <- c(0.1,0.5,1.0,1.5,2.0,3.0)
  smb.rbf.nlms.res <- matrix(0,nrow=length(smb.rbf.nlms.lrs),ncol=nbr.itrs)
  for ( lrindex in 1:length(smb.rbf.nlms.lrs) ) {
    ol$set.lr(smb.rbf.nlms.lrs[lrindex])
    ol$set.weights(matrix(rep(0, nbr.rbf.grid.x*nbr.rbf.grid.y),nrow=1))
    for ( iteration in 1:nbr.itrs ) {
      train.whole.set(t.in, targets, rbf, iterations=1)
      ty <- rbf$eval(t.in)
      smb.rbf.nlms.res[lrindex,iteration] <- root.mean.squared.error(targets,ty)
    }
  }
  
  #  "Ordinary" NRBF
  rbf$set.nrbf(TRUE)
  ol$set.use.bias(FALSE)
  ol$set.nlms(FALSE)
  smb.nrbf.lrs <- c(1,5,10,50,100,200)
  smb.nrbf.res <- matrix(0,nrow=length(smb.nrbf.lrs),ncol=nbr.itrs)
  for ( lrindex in 1:length(smb.nrbf.lrs) ) {
    ol$set.lr(smb.nrbf.lrs[lrindex])
    ol$set.weights(matrix(rep(0, nbr.rbf.grid.x*nbr.rbf.grid.y),nrow=1))
    for ( iteration in 1:nbr.itrs ) {
      train.whole.set(t.in, targets, rbf, iterations=1)
      ty <- rbf$eval(t.in)
      smb.nrbf.res[lrindex,iteration] <- root.mean.squared.error(targets,ty)
    }
  }
  
  # NRBF with NLMS
  rbf$set.nrbf(TRUE)
  ol$set.use.bias(FALSE)
  ol$set.nlms(TRUE)
  smb.nrbf.nlms.lrs <- c(0.1,0.5,1.0,1.5,2.0,3.0)
  smb.nrbf.nlms.res <- matrix(0,nrow=length(smb.nrbf.nlms.lrs),ncol=nbr.itrs)
  for ( lrindex in 1:length(smb.nrbf.nlms.lrs) ) {
    ol$set.lr(smb.nrbf.nlms.lrs[lrindex])
    ol$set.weights(matrix(rep(0, nbr.rbf.grid.x*nbr.rbf.grid.y),nrow=1))
    for ( iteration in 1:nbr.itrs ) {
      train.whole.set(t.in, targets, rbf, iterations=1)
      ty <- rbf$eval(t.in)
      smb.nrbf.nlms.res[lrindex,iteration] <- root.mean.squared.error(targets,ty)
    }
  }
  assign("smb.rbf.lrs", smb.rbf.lrs, pos=.GlobalEnv)
  assign("smb.rbf.nlms.lrs", smb.rbf.nlms.lrs, pos=.GlobalEnv)
  assign("smb.nrbf.lrs", smb.nrbf.lrs, pos=.GlobalEnv)
  assign("smb.nrbf.nlms.lrs", smb.nrbf.nlms.lrs, pos=.GlobalEnv)
  assign("smb.rbf.res", smb.rbf.res, pos=.GlobalEnv)
  assign("smb.rbf.nlms.res", smb.rbf.nlms.res, pos=.GlobalEnv)
  assign("smb.nrbf.res", smb.nrbf.res, pos=.GlobalEnv)
  assign("smb.nrbf.nlms.res", smb.nrbf.nlms.res, pos=.GlobalEnv)
  #print(smb.rbf.res)
  #print(smb.rbf.nlms.res)
  #print(smb.nrbf.res)
  #print(smb.nrbf.nlms.res)

  # Best learning rates are (in order): 0.1, 1.0, 100, 1.0
  
  ol$set.lr(1)
  rbf$set.nrbf(TRUE)
  ol$set.nlms(TRUE)
  ol$set.use.bias(FALSE)
  ol$set.weights(matrix(rep(0, nbr.rbf.grid.x*nbr.rbf.grid.y),nrow=1))
  nbr.itrs <- 50
  train.whole.set(t.in, targets, rbf, iterations=nbr.itrs)
  xstep <- 0.21
  ystep <- 0.21
  xseq <- seq(xmin, xmax, xstep)
  yseq <- seq(ymin, ymax, ystep)
  xvals <- rep(xseq, length(yseq))
  yvals <- rep(yseq, each=length(xseq))
  zvals <- rbf$eval(matrix(c(xvals,yvals),ncol=2))
  z <- matrix(zvals, nrow=length(xseq), ncol=length(yseq))
  persp(xseq, yseq, z)
}

nlms.vs.nrbf.iris <- function() {
  # 4 input variables, three possible Iris classes
  n.in <- 4
  n.out <- 3
  rbf <- rbf.new(n.in, n.out, 0, activation.function=squared.distance.activation, output.function=imqe.output.function)
  t.in <- as.matrix(iris[,1:4])
  in.mins <- apply(t.in, 2, min)
  in.maxs <- apply(t.in, 2, max)
  setosas <- (iris[,5]=="setosa")*1
  versicolors <- (iris[,5]=="versicolor")*1
  virginicas <- (iris[,5]=="virginica")*1
  targets <- matrix(c(setosas,versicolors,virginicas), ncol=3)

  # Initialise hidden layer. Normalise input matrix to [0,1].
  aff.trans <- scale.translate.ranges(in.mins, in.maxs, c(0,0,0,0), c(1,1,1,1))
  nrbf <- 3 # Number of neurons per dimension, gives nrbf^4 neurons
  hl <- rbf$get.hidden()
  hl$init.centroids.grid(in.mins, in.maxs, c(nrbf,nrbf,nrbf,nrbf),affine.transformation=aff.trans)
  ol <- rbf$get.outlayer()
  ol$set.use.bias(FALSE)
  ol$set.weights(matrix(rep(0, nrbf^n.in*n.out),nrow=3))
  rbf$set.spread(0.01)

  # Set up experiments
  nbr.itrs <- 500

  # RBF, no normalisation.
  iris.rbf.lrs <- c(0.001,0.005,0.01,0.05,0.1,0.5)
  iris.rbf.res <- matrix(0,nrow=length(iris.rbf.lrs),ncol=nbr.itrs)
  rbf$set.nrbf(FALSE)
  ol$set.use.bias(FALSE)
  ol$set.nlms(FALSE)
  for ( lrindex in 1:length(iris.rbf.lrs) ) {
    ol$set.lr(iris.rbf.lrs[lrindex])
    ol$set.weights(matrix(rep(0, nrbf^n.in*n.out),nrow=3))
    for ( iteration in 1:nbr.itrs ) {
      train.whole.set(t.in, targets, rbf, iterations=1)
      ty <- rbf$eval(t.in)
      #iris.nrbf.nlms.res[lrindex,iteration] <- root.mean.squared.error(targets,ty) # RMSE is not very good measure here.
      classification <- (ty == apply(ty, 1, max))*1
      iris.rbf.res[lrindex,iteration] <- sum(abs(targets - classification))/2
    }
  }

  # RBF, NLMS. 
  iris.rbf.nlms.lrs <- c(0.001,0.005,0.01,0.05,0.1,0.5)
  iris.rbf.nlms.res <- matrix(0,nrow=length(iris.rbf.nlms.lrs),ncol=nbr.itrs)
  rbf$set.nrbf(FALSE)
  ol$set.use.bias(FALSE)
  ol$set.nlms(TRUE)
  for ( lrindex in 1:length(iris.rbf.nlms.lrs) ) {
    ol$set.lr(iris.rbf.nlms.lrs[lrindex])
    ol$set.weights(matrix(rep(0, nrbf^n.in*n.out),nrow=3))
    for ( iteration in 1:nbr.itrs ) {
      train.whole.set(t.in, targets, rbf, iterations=1)
      ty <- rbf$eval(t.in)
      #iris.nrbf.nlms.res[lrindex,iteration] <- root.mean.squared.error(targets,ty) # RMSE is not very good measure here.
      classification <- (ty == apply(ty, 1, max))*1
      iris.rbf.nlms.res[lrindex,iteration] <- sum(abs(targets - classification))/2
    }
  }
 
  #  "Ordinary" NRBF
  iris.nrbf.lrs <- c(0.1,0.5,1.0,1.5,2.0,3.0,4.0)
  iris.nrbf.res <- matrix(0,nrow=length(iris.nrbf.lrs),ncol=nbr.itrs)
  rbf$set.nrbf(TRUE)
  ol$set.use.bias(FALSE)
  ol$set.nlms(FALSE)
  for ( lrindex in 1:length(iris.nrbf.lrs) ) {
    ol$set.lr(iris.nrbf.lrs[lrindex])
    ol$set.weights(matrix(rep(0, nrbf^n.in*n.out),nrow=3))
    for ( iteration in 1:nbr.itrs ) {
      train.whole.set(t.in, targets, rbf, iterations=1)
      ty <- rbf$eval(t.in)
      classification <- (ty == apply(ty, 1, max))*1
      iris.nrbf.res[lrindex,iteration] <- sum(abs(targets - classification))/2
    }
  }
  
  # NRBF with NLMS
  iris.nrbf.nlms.lrs <- c(0.01,0.05,0.1,0.5,1.0)
  iris.nrbf.nlms.res <- matrix(0,nrow=length(iris.nrbf.nlms.lrs),ncol=nbr.itrs)
  rbf$set.nrbf(TRUE)
  ol$set.use.bias(FALSE)
  ol$set.nlms(TRUE)
  for ( lrindex in 1:length(iris.nrbf.nlms.lrs) ) {
    ol$set.lr(iris.nrbf.nlms.lrs[lrindex])
    ol$set.weights(matrix(rep(0, nrbf^n.in*n.out),nrow=3))
    for ( iteration in 1:nbr.itrs ) {
      train.whole.set(t.in, targets, rbf, iterations=1)
      ty <- rbf$eval(t.in)
      #iris.nrbf.nlms.res[lrindex,iteration] <- root.mean.squared.error(targets,ty) # RMSE is not very good measure here.
      classification <- (ty == apply(ty, 1, max))*1
      iris.nrbf.nlms.res[lrindex,iteration] <- sum(abs(targets - classification))/2
    }
  }

  assign("iris.rbf.lrs", iris.rbf.lrs, pos=.GlobalEnv)
  assign("iris.rbf.nlms.lrs", iris.rbf.nlms.lrs, pos=.GlobalEnv)
  assign("iris.nrbf.lrs", iris.nrbf.lrs, pos=.GlobalEnv)
  assign("iris.nrbf.nlms.lrs", iris.nrbf.nlms.lrs, pos=.GlobalEnv)
  assign("iris.rbf.res", iris.rbf.res, pos=.GlobalEnv)
  assign("iris.rbf.nlms.res", iris.rbf.nlms.res, pos=.GlobalEnv)
  assign("iris.nrbf.res", iris.nrbf.res, pos=.GlobalEnv)
  assign("iris.nrbf.nlms.res", iris.nrbf.nlms.res, pos=.GlobalEnv)
  #print(iris.rbf.res)
  #print(iris.rbf.nlms.res)
  #print(iris.nrbf.res)
  #print(iris.nrbf.nlms.res)
  
  # Initial calculations that worked fine.
  ol$set.lr(0.1)
  rbf$set.nrbf(TRUE)
  ol$set.nlms(TRUE)
  ol$set.weights(matrix(rep(0, nrbf^n.in*n.out),nrow=3))
  nbr.itrs <- 200
  train.whole.set(t.in, targets, rbf, iterations=nbr.itrs)
  #for ( row in 1:nrow(t.in) ) {
  #  print(rbf$eval(t.in[row,]))
  #}
  res <- rbf$eval(t.in)
  #print(res)                 
  #print((res == apply(res, 1, max))*1)                 
  print(colSums(res == apply(res, 1, max)))                 
}

# Best results with following parameters:
# - smb.rbf: 0.1 (ind=3). Sum: 4.323913. Final: 0.007185862
# - smb.rbf.nlms: 1.0 (ind=3). Sum: 4.29109. Final: 0.007345287
# - smb.nrbf: 100 (ind=5). Sum: 4.856243. Final: 0.00703767
# - smb.nrbf.nlms: 1.0 (ind=3). Sum: 4.639341. Final: 0.006757741

# - iris.rbf: 0.01 (ind=3). 4 misclassified on 48th episode
# - iris.rbf.nlms: 0.050 (ind=4). 4 misclassified on 40th episode
# - iris.nrbf: 2.0 (ind=5). 4 misclassified on 34th episode. Only 3 misclassified during episodes 40-54.
# - iris.nrbf.nlms: 0.05 (ind=2). 4 misclassified on 30th episode. 

