# Everything needed for creating Mountain Car results and graphs
source("Functions.R")
source("NeuralLayer.R")
source("Adaline.R")
source("SarsaLearner.R")
source("BIMM.R")
source("EligibilityTrace.R")
source("LinearWalk.R")
source("MountainCar.R")
source("DiscreteClassifier.R")
source("CMAC.R")
source("RBFclassifier.R")

get.resvar.names <- function(var.prefix, alphas, lambdas) {
  resvars <- vector("character", length=length(alphas))
  for ( i in 1:length(alphas) ) {
    resvars[i] <- paste(var.prefix, "a", alphas[i], "_l", lambdas[i], sep="")
  }
  return(resvars) 
}

run.mc.agents <- function(nagents, neps, classifier, alphas, lambdas, accumulating.trace=FALSE, trace.reset=TRUE, var.prefix="",dr=1.0,epsilon=0.0,step.reward=-1,goal.reward=0) {
  resvars <- get.resvar.names(var.prefix, alphas, lambdas)
  for ( i in 1:length(alphas) ) {
    res <- sarsa.mc(nagents,neps,classifier=classifier,lr=alphas[i],dr=dr,lambda=lambdas[i],epsilon,step.reward,goal.reward,accumulating.trace=accumulating.trace,trace.reset=trace.reset,visualize=FALSE)
    assign(resvars[i], res, pos=.GlobalEnv)
    print(paste("Finished calculating", resvars[i]))
  }
  return(resvars)
}

# All variables print out all relevant comparative statistics
get.standard.error <- function(datamatrix) {
  v <- datamatrix
  dim(v) <- c(nrow(datamatrix)*ncol(datamatrix), 1)
  stde <- sd(v)/sqrt(nrow(v))
}

print.comparison <- function(lambdas, alphas, res.with.reset, res.no.reset) {
  for ( i in 1:length(lambdas) ) {
    v1 <- get(res.with.reset[i])
    v2 <- get(res.no.reset[i])
    m1 <- mean(v1)
    m2 <- mean(v2)
    v <- v1; dim(v) <- c(nrow(v1)*ncol(v1), 1)
    se1 <- sd(v)/sqrt(nrow(v))
    v <- v2; dim(v) <- c(nrow(v1)*ncol(v2), 1)
    se2 <- sd(v)/sqrt(nrow(v))
    s <- paste("lambda: ", lambdas[i], " lr: ", alphas[i], " mean with reset: ", m1, ", mean without reset: ", m2, " SE(reset): ", se1, " SE(no reset): ", se2, sep="")
    print(s)
  }
}

plot.comparison <- function(x, r1, r2) {
  m1 <- vector(mode="numeric", length=length(r1))
  m2 <- vector(mode="numeric", length=length(r2))
  se1 <- vector(mode="numeric", length=length(r1))
  se2 <- vector(mode="numeric", length=length(r2))
  for ( i in 1:length(r1) ) {
    v1 <- get(r1[i])
    v2 <- get(r2[i])
    m1[i] <- mean(v1)
    m2[i] <- mean(v2)
    v <- v1; dim(v) <- c(nrow(v1)*ncol(v1), 1)
    se1[i] <- sd(v)/sqrt(nrow(v))
    v <- v2; dim(v) <- c(nrow(v1)*ncol(v2), 1)
    se2[i] <- sd(v)/sqrt(nrow(v))
  }
  # Empty plot with correct x/y ranges
  xrange <- c(min(x), max(x))
  yrange <- c(min(m1, m2, m1-se1, m2-se2), max(m1, m2, m1+se1, m2+se2)) 
  plot(xrange, yrange, type="n", xlab=expression(lambda), ylab="Steps/Trial. Averaged over first 20 trials and 50 runs")
  points(m2~x)
  points(m1~x, pch=22)
  arrows(x, m1-se1, x, m1+se1, code=3, angle=90, length=0.1)
  arrows(x, m2-se2, x, m2+se2, code=3, angle=90, length=0.1, col="green")
}

# Parameter "experiment.nbr" indicates what labels to include in the
# graph. If it has zero value, labels are put into default positions.
plot.lines <- function(names, alphas, lambdas, xrange=NULL, yrange=NULL, leftlabel="", experiment.nbr=0) {
  # Always create new device so that we get the size we want
  if ( experiment.nbr == 5 )
    x11(width=8,height=5)
  else
    x11(width=11,height=5)
  
  # Create matrices for containing all values to plot
  m <- list()
  n <- list()
  v <- list()
  
  for ( nind in 1:length(names) ) {
    m[[nind]] <- matrix(nrow=length(lambdas),ncol=length(alphas))
    # For all "lambda" values, we draw two lines
    for ( lambda in 1:length(lambdas) ) {
      for ( alpha in 1:length(alphas) ) {
        n[[nind]] <- get.resvar.names(names[nind], alphas[alpha],lambdas[lambda])
        if ( exists(n[[nind]]) ) {
          v[[nind]] <- get(n[[nind]])
          m[[nind]][lambda,alpha] <- mean(v[[nind]])
        }
        else
          m[[nind]][lambda,alpha] <- NA # Should get this from MountainCar but to lazy just now...
      }
    }
  }
    
  # Empty plots with correct x/y ranges.
  oldpar <- par(no.readonly=TRUE)
  if ( experiment.nbr == 5 )
    par(mfcol=c(1,2))
  else
    par(mfcol=c(1,3))
  if ( is.null(xrange) ) 
    xrange <- c(min(alphas), max(alphas))
  if ( is.null(yrange) ) 
    yrange <- c(min(m1, m2), max(m1, m2))
  par(mar=c(5,2.5,4,1))

  #
  # Accumulating trace
  #
  plot(xrange, yrange, type="n", xlab=expression(alpha), ylab=leftlabel, yaxt="n", main="Accumulating trace")
  axis(4, labels=FALSE)
  mtext(leftlabel, side=2, line=1)
  for ( lambda in 1:length(lambdas) ) {
    lines(m[[1]][lambda,]~alphas, lty=lambda)
    points(m[[1]][lambda,]~alphas, lty=lambda)
    # Default labels. Special case for experiment 3!
    if ( experiment.nbr == 0 || experiment.nbr == 3 || experiment.nbr == 4 )
      text(alphas[1], m[[1]][lambda,1], lambdas[lambda], pos=4)
  }

  # Add labels depending on the plot. Here: discounted, lookup-table
  #if ( experiment.nbr == 2 ) {
  #  text(alphas[4], m2[1,4], lambdas[1], pos=4)
  #  text(alphas[1], m2[2,1], lambdas[2], pos=4)
  #  text(alphas[5], m2[3,5], lambdas[3], pos=3)
  #  text(alphas[length(alphas)], m2[4,length(alphas)], lambdas[4], pos=2)
  #  text(alphas[2], m2[5,2], lambdas[5], pos=4)
  #  text(alphas[5], m2[6,5], lambdas[6], pos=1)
  #  text(alphas[4], m2[7,4], lambdas[7], pos=3)
  #}

  # Add labels depending on the plot. Here: OIV, CMAC
  #if ( experiment.nbr == 3 ) {
  #  text(alphas[length(alphas)]-0.005, m2[1,length(alphas)]+10, lambdas[1], pos=3)
  #}
  
  # Add labels depending on the plot. Here: discounted, CMAC
  #if ( experiment.nbr == 4 ) {
  #  text(alphas[4], m2[1,4], lambdas[1], pos=2)
  #}

  # Add labels depending on the plot. Here: OIV, NRBF
  if ( experiment.nbr == 5 ) {
    text(alphas[6], m[[1]][1,6], lambdas[1], pos=4)
    text(alphas[17], m[[1]][2,17], lambdas[2], pos=1)
    text(alphas[13], m[[1]][3,13], lambdas[3], pos=3)
    text(alphas[2], m[[1]][6,2], lambdas[6], pos=1)
  }
  
  if ( experiment.nbr != 5 ) {
    plot(xrange, yrange, type="n", xlab=expression(alpha), main="With reset")
    axis(4, labels=FALSE)
    for ( lambda in 1:length(lambdas) ) {
      lines(m[[2]][lambda,]~alphas, lty=lambda)
      points(m[[2]][lambda,]~alphas, lty=lambda)
      # Default labels.
      if ( experiment.nbr == 0 )
        text(alphas[length(alphas)], m[[2]][lambda,length(alphas)]+10, lambdas[lambda], pos=2)

    }
  
    # Add labels depending on the plot. Here: discounted, lookup-table
    if ( experiment.nbr == 2 ) {
      text(alphas[3], m[[2]][1,3], lambdas[1], pos=2)
      text(alphas[length(alphas)], m[[2]][2,length(alphas)]+10, lambdas[2], pos=2)
      text(alphas[4], m[[2]][3,4], lambdas[3], pos=2)
      text(alphas[2]+0.02, m[[2]][4,2], lambdas[4], pos=3)
      text(alphas[4], m[[2]][5,4], lambdas[5], pos=3)
      text(alphas[length(alphas)]+0.03, m[[2]][6,length(alphas)]-25, lambdas[6], pos=2)
      text(alphas[5], m[[2]][7,5], lambdas[7], pos=1)
    }
  
    # Add labels depending on the plot. Here: OIV, CMAC
    if ( experiment.nbr == 3 ) {
      text(alphas[7], m[[2]][1,7], lambdas[1], pos=2)
      text(alphas[2], m[[2]][2,2], lambdas[2], pos=3)
      text(alphas[1], m[[2]][3,1], lambdas[3], pos=3)
      text(alphas[6], m[[2]][4,6], lambdas[4], pos=1)
      text(alphas[2]-0.005, m[[2]][5,2], lambdas[5], pos=4)
      text(alphas[3]-0.005, m[[2]][6,3], lambdas[6], pos=4)
      text(alphas[5], m[[2]][7,5], lambdas[7], pos=1)
    }
  
    # Add labels depending on the plot. Here: discounted, CMAC
    if ( experiment.nbr == 4 ) {
      text(alphas[5], m[[2]][1,5]+100, lambdas[1], pos=3)
      text(alphas[2], m[[2]][2,2], lambdas[2], pos=2)
      text(alphas[3], m[[2]][3,3], lambdas[3], pos=3)
      text(alphas[2], m[[2]][4,2], lambdas[4], pos=3)
      text(alphas[4], m[[2]][5,4], lambdas[5], pos=3)
      text(alphas[4], m[[2]][6,4], lambdas[6], pos=4)
      text(alphas[2], m[[2]][7,2], lambdas[7], pos=1)
    }
  }
  
  plot(xrange, yrange, type="n", xlab=expression(alpha), main="Replacing trace")
  for ( lambda in 1:length(lambdas) ) {
    lines(m[[3]][lambda,]~alphas, lty=lambda)
    points(m[[3]][lambda,]~alphas, lty=lambda)
    # Default labels. Special case for experiment 3!
    if ( experiment.nbr == 0 || experiment.nbr == 3 || experiment.nbr == 4 )
      text(alphas[1], m[[3]][lambda,1], lambdas[lambda], pos=4)
  }

  # Add labels depending on the plot. Here: discounted, lookup-table
  if ( experiment.nbr == 2 ) {
    text(alphas[4], m[[3]][1,4], lambdas[1], pos=4)
    text(alphas[1], m[[3]][2,1], lambdas[2], pos=4)
    text(alphas[5], m[[3]][3,5], lambdas[3], pos=3)
    text(alphas[length(alphas)], m[[3]][4,length(alphas)], lambdas[4], pos=2)
    text(alphas[2], m[[3]][5,2], lambdas[5], pos=4)
    text(alphas[5], m[[3]][6,5], lambdas[6], pos=1)
    text(alphas[4], m[[3]][7,4], lambdas[7], pos=3)
  }

  # Add labels depending on the plot. Here: OIV, CMAC
  if ( experiment.nbr == 3 ) {
    text(alphas[length(alphas)]-0.005, m[[3]][1,length(alphas)]+10, lambdas[1], pos=3)
  }
  
  # Add labels depending on the plot. Here: discounted, CMAC
  if ( experiment.nbr == 4 ) {
    text(alphas[4], m[[3]][1,4], lambdas[1], pos=2)
  }
  
  # Add labels depending on the plot. Here: OIV, NRBF
  if ( experiment.nbr == 5 ) {
    text(alphas[6], m[[3]][1,6], lambdas[1], pos=4)
    text(alphas[3], m[[3]][7,3], lambdas[7], pos=1)
  }
  
  par(oldpar)
}

# Parameter "experiment.nbr" indicates what labels to include in the
# graph. If it has zero value, labels are put into default positions.
plot.final.comparison <- function(names, lambdas, best.alphas, leftlabel="", experiment.nbr=0, min.y=NULL, max.y=NULL, main.title="", nbr.plot.agents=NULL, nbr.plot.episodes=NULL) {

  # Always create new device so that we get the size we want
  x11(width=5,height=5)

  # We assume the number of names indicates how many data series we have
  nbr.names <- length(names)
  
  # Create all matrices needed
  m <- list();
  se <- list()
  n <- list()
  v <- list()
  for ( nind in 1:nbr.names ) {
    m[[nind]] <- vector(mode="numeric", length=length(lambdas))
    se[nind] <- vector(mode="numeric", length=length(lambdas))
    for ( i in 1:length(lambdas) ) {
      n[[nind]] <- get.resvar.names(names[nind],best.alphas[[nind]][i],lambdas[i])
      if ( exists(n[[nind]]) ) {
        v[[nind]] <- get(n[[nind]])
        if ( !is.null(nbr.plot.agents) )
          v[[nind]] <- v[[nind]][1:min(nbr.plot.agents,nrow(v[[nind]])),]
        if ( !is.null(nbr.plot.episodes) )
          v[[nind]] <- v[[nind]][,1:min(nbr.plot.episodes,ncol(v[[nind]]))]
        m[[nind]][i] <- mean(v[[nind]])
        x <- v[[nind]]; dim(x) <- c(nrow(v[[nind]])*ncol(v[[nind]]), 1)
        se[[nind]][i] <- sd(x)/sqrt(nrow(x))
      }
      else {
        m[[nind]][i] <- NA
        se[[nind]][i] <- NA
      }
    }
  }

  # Set minimum and maximum y-scale values unless they are given as arguments. 
  if ( is.null(min.y) ) {
    min.y <- min(m[[1]] )
    for ( i in 2:nbr.names) 
      min.y <- min(min.y,m[[i]])
  }
  if ( is.null(max.y) ) {
    max.y <- max(m[[1]])
    for ( i in 2:nbr.names ) 
      max.y <- max(max.y,m[[i]])
  }

  # Empty plot with correct x/y ranges
  xrange <- c(min(lambdas), max(lambdas))
  yrange <- c(min.y, max.y)
  plot(xrange, yrange, type="n", main=main.title, xlab=expression(lambda), ylab=leftlabel)
  for ( i in 1:nbr.names ) {
    # Temporary quick solution here, not good!
    if ( i == 1 )
      pch <- 18
    else if ( i == 2 ) 
      pch <- 1
    else if ( i == 3 ) 
      pch <- 2
    else
      pch <- i
    if ( !is.null(m[[i]]) ) {
      lines(m[[i]]~lambdas, lty=i)
      points(m[[i]]~lambdas, pch=pch)
      arrows(lambdas, m[[i]]-se[[i]], lambdas, m[[i]]+se[[i]], code=3, angle=90, length=0.1)
    }
  }

  # Add labels depending on the plot. Here: OIV, lookup-table
  if ( experiment.nbr == 1 ) {
    text(lambdas[4]-0.1, m[[1]][4], "accumulating", pos=2)
    arrows(lambdas[4]-0.1,m[[1]][4],lambdas[3]+(lambdas[4]-lambdas[3])/2,m[[1]][3]+(m[[1]][4]-m[[1]][3])/2,length=0.1)
    text(lambdas[1]+(lambdas[2]-lambdas[1])/2, m[[2]][2]-10, "with reset", pos=1)
    arrows(lambdas[1]+(lambdas[2]-lambdas[1])/2,m[[2]][2]-10,lambdas[1]+(lambdas[2]-lambdas[1])/2,m[[2]][1]+(m[[2]][2]-m[[2]][1])/2,length=0.1)
    text(lambdas[2]+0.2, m[[3]][2], "without reset", pos=3)
    arrows(lambdas[2]+0.2,m[[3]][2],lambdas[2]+(lambdas[3]-lambdas[2])/2,m[[3]][2]+(m[[3]][3]-m[[3]][2])/2,length=0.1)
  }

  # Add labels depending on the plot. Here: discounted, lookup-table
  if ( experiment.nbr == 2 ) {
    text(lambdas[2], m[[1]][2], "accumulating", pos=1)
    text(lambdas[3], m[[2]][3]+10, "with reset", pos=4)
    text(lambdas[5], m[[3]][5], "without reset", pos=2)
  }

  # Add labels depending on the plot. Here: OIV, CMAC
  if ( experiment.nbr == 3 ) {
    text(lambdas[6]-0.05, m[[1]][6]+2, "accumulating", pos=3)
    text(lambdas[2], m[[2]][2]+1, "with reset", pos=4)
    text(lambdas[3], m[[3]][3], "without reset", pos=2)
  }

  # Add labels depending on the plot. Here: discounted, CMAC
  if ( experiment.nbr == 4 ) {
    text(lambdas[2], m[[1]][2], "accumulating", pos=1)
    text(lambdas[2], m[[2]][2], "with reset", pos=4)
    text(lambdas[2], m[[3]][2], "without reset", pos=2)
  }

  # Add labels depending on the plot. Here: OIV, RBF
  if ( experiment.nbr == 5 ) {
    text(lambdas[7], m[[1]][7], "accumulating", pos=2)
    text(lambdas[5], m[[3]][5], "replacing", pos=2)
  }

  # Add labels depending on the plot. Here: CMAC, RBF for 30 agents, 20 eps
  # as in Singh&Sutton
  if ( experiment.nbr == 6 ) {
    text(lambdas[6], m[[1]][6], "CMAC accumulating", pos=2)
    text(lambdas[6], m[[2]][6], "CMAC replacing with reset", pos=2)
    text(lambdas[6], m[[3]][6], "RBF accumulating", pos=2)
    text(lambdas[3], m[[4]][3], "RBF replacing without reset", pos=2)
  }

}

# Call this function to re-generate all results. 
calculate.all <- function(prefix.names, nbr.agents, nbr.episodes, classifier, alphas, lambdas,dr=1.0,epsilon=0.0,step.reward=-1,goal.reward=0) {
  all.alphas <- rep(alphas, each=length(lambdas))
  all.lambdas <- rep(lambdas, times=length(alphas))
  #assign("accumulating", run.mc.agents(nbr.agents, nbr.episodes, classifier, all.alphas, all.lambdas, accumulating.trace=TRUE, trace.reset=FALSE, var.prefix=prefix.names[1],dr=dr,epsilon=epsilon,step.reward=step.reward,goal.reward=goal.reward), pos=.GlobalEnv)
  #assign("with.reset", run.mc.agents(nbr.agents, nbr.episodes, classifier, all.alphas, all.lambdas, trace.reset=TRUE, var.prefix=prefix.names[2],dr=dr,epsilon=epsilon,step.reward=step.reward,goal.reward=goal.reward), pos=.GlobalEnv)
  assign("no.reset", run.mc.agents(nbr.agents, nbr.episodes, classifier, all.alphas, all.lambdas, trace.reset=FALSE, var.prefix=prefix.names[3],dr=dr,epsilon=epsilon,step.reward=step.reward,goal.reward=goal.reward), pos=.GlobalEnv)
}

# Prints out averages for "eps" last episodes
print.last.averages <- function(names, lambdas, alphas, eps) {
  for ( name in 1:length(names) ) {
    for ( i in 1:length(lambdas) ) {
      n <- get.resvar.names(names[name],alphas[name,i],lambdas[i])
      v <- get(n); nc <- ncol(v);
      m <- mean(v[,(nc-eps+1):nc]); 
      print(paste("Average steps for last", eps, "episodes,", n, ":", m))
    }
  }
}

#print.comparison(lambdas, alphas, reset.results, no.reset.results)
#plot.comparison(lambdas, reset.results, no.reset.results)

#======================================================================
# Set up classifier, which can be DiscreteClassifier or CMAC. Or
# something else in the future...
#
xints <- 8
vints <- 8
nbr.layers <- 5
classifier <- discrete.classifier.new(nbr.dimensions=2, minvals=c(-1.2,-0.07), maxvals=c(0.5,0.07), nbr.classes=c(xints,vints))

#======================================================================
# Next section contains calls for repeating the experiments in
# Singh&Sutton with 8x8 discretisation
#
STR.SUTTON.SINGH.RESET <- "reset."
STR.SUTTON.SINGH.NO.RESET <- "no.reset."
STR.SUTTON.SINGH.ACCUMULATING <- "accumulating."
names <- c(STR.SUTTON.SINGH.ACCUMULATING, STR.SUTTON.SINGH.RESET, STR.SUTTON.SINGH.NO.RESET)

nbr.agents <- 50 # 20 more than in Singh&Sutton (30)
nbr.episodes <- 20 # As in Singh&Sutton (20)
xints <- 8
vints <- 8
alphas <- c(0.1, 0.3, 0.5, 0.7, 0.9)
lambdas <- c(0, 0.4, 0.7, 0.8, 0.9, 0.95, 0.99)
# Calculate all results
#calculate.all(names, nbr.agents, nbr.episodes, classifier, alphas, lambdas)

# Or load them. But this also loads all old versions of functions,
# which can give bugs!
#load("D:/KF/Software/R/KFnnet/Icann06Runs.RData") 

# First plot: all alphas, lambdas
#plot.lines(names, alphas, lambdas, yrange=c(200,1000), leftlabel="Steps/Episode. Averaged over first 20 episodes and 50 runs")

# Second plot: best results, with standard error
best.alphas.reset <- c(0.7,0.5,0.7,0.7,0.7,0.5,0.7)
best.alphas.no.reset <- c(0.5,0.5,0.3,0.5,0.5,0.7,0.5)
best.lambdas <- lambdas
#plot.final.comparison(names, lambdas, best.alphas.reset, best.alphas.no.reset, leftlabel=expression(paste("Average Steps/Episode at best ", alpha)))

#======================================================================
# Next section contains calls for doing experiments in
# Singh&Sutton with 8x8 discretisation and 200 episodes instead of 30
#
STR.SUTTON.SINGH.RESET.200E <- "reset.200e."
STR.SUTTON.SINGH.NO.RESET.200E <- "no.reset.200e."
STR.SUTTON.SINGH.ACCUMULATING.200E <- "acc.200e."
names <- c(STR.SUTTON.SINGH.ACCUMULATING.200E, STR.SUTTON.SINGH.RESET.200E, STR.SUTTON.SINGH.NO.RESET.200E)

nbr.agents <- 50 # 20 more than in Singh&Sutton (30)
nbr.episodes <- 200 # More than Singh&Sutton (20)
xints <- 8
vints <- 8
alphas <- c(0.05, 0.1, 0.3, 0.5, 0.7, 0.9)
lambdas <- c(0, 0.4, 0.7, 0.8, 0.9, 0.95, 0.99)
#alphas <- c(0.3,0.9)
#lambdas <- c(0.95)
# Calculate all results
# Accumulating trace with with 0.9 and greater diverges.
# From alpha >= 0.7 already lambda 0.4 diverges.
# From alpha >= 0.5 already lambda 0.7 diverges.
# Action-value function becomes completely erroneous.
#calculate.all(names, nbr.agents, nbr.episodes, classifier, alphas, lambdas)

# Or load them. But this also loads all old versions of functions,
# which can give bugs!
#load("D:/KF/Software/R/KFnnet/MC_OIV_LookupTable.RData") 

# First plot: all alphas, lambdas
#plot.lines(names, alphas, lambdas, yrange=c(100,600), leftlabel="Steps/Episode. Averaged over first 200 episodes and 50 runs")

# Second plot: best results, with standard error
best.alphas.acc <- c(0.3,0.3,0.1,0.1,0.1,0.05,0.05)
best.alphas.reset <- c(0.3,0.3,0.3,0.3,0.3,0.3,0.3)
best.alphas.no.reset <- c(0.3,0.3,0.3,0.3,0.3,0.3,0.3)
best.alphas <- list(a=best.alphas.acc,b=best.alphas.reset,c=best.alphas.no.reset)
best.lambdas <- lambdas
#plot.final.comparison(names, lambdas, best.alphas, leftlabel=expression(paste("Average Steps/Episode at best ", alpha)), experiment.nbr=1, main.title="Mountain-Car, Lookup-Table")

#======================================================================
# Next section contains calls for doing experiments in
# Singh&Sutton with 8x8 discretisation and discounted reward and
# zero step reward, 1 goal reward.
#
MC.RESET.D09.100E <- "reset.discounted.100e."
MC.NO.RESET.D09.100E <- "no.reset.discounted.100e."
MC.ACC.D09.100E <- "acc.discounted.100e."
names <- c(MC.ACC.D09.100E, MC.RESET.D09.100E, MC.NO.RESET.D09.100E)

nbr.agents <- 50 # 20 more than in Singh&Sutton (30)
nbr.episodes <- 100 # More than Singh&Sutton (20)
xints <- 8
vints <- 8
dr <- 0.9
#alphas <- c(0.05)
alphas <- c(0.05, 0.1, 0.3, 0.5, 0.7, 0.9)
lambdas <- c(0, 0.4, 0.7, 0.8, 0.9, 0.95, 0.99)
#lambdas <- c(0.9)
# Calculate all results
#calculate.all(names, nbr.agents, nbr.episodes, classifier, alphas, lambdas, dr=dr,epsilon=0.1,step.reward=0,goal.reward=1)

# Or load them. But this also loads all old versions of functions,
# which can give bugs!
#load("D:/KF/Software/R/KFnnet/MC_discountedRuns.RData") 

# First plot: all alphas, lambdas
#plot.lines(names, alphas, lambdas, yrange=c(300,1000), leftlabel="Steps/Episode. Averaged over first 100 episodes and 50 runs", experiment.nbr=2)

# Second plot: best results, with standard error
#best.alphas.reset <- c(0.3,0.3,0.3,0.3,0.3,0.3,0.3)
#best.alphas.no.reset <- c(0.3,0.3,0.3,0.3,0.3,0.3,0.3)
best.alphas.reset <- c(0.3,0.3,0.3,0.3,0.1,0.05,0.1)
best.alphas.no.reset <- c(0.5,0.1,0.3,0.05,0.1,0.7,0.3)
best.lambdas <- lambdas
#plot.final.comparison(names, lambdas, best.alphas.reset, best.alphas.no.reset, leftlabel=expression(paste("Average Steps/Episode at best ", alpha)), experiment.nbr=2)

#======================================================================
# From here on we have CMAC tests. Classifier generation has been
# commented so that all simulations use the same classifier. Otherwise
# the random shifts of discretisations may give unexpected differences. 
#
classifier <- cmac.new(nbr.layers=nbr.layers, nbr.dimensions=2, minvals=c(-1.2,-0.07), maxvals=c(0.5,0.07), nbr.classes=c(xints,vints), random.offset=FALSE)

#======================================================================
# Next section contains calls for 8x8 CMAC experiments in
# Singh&Sutton and 100 episodes instead of 30
#
names <- c("cmac.acc.100e.", "cmac.reset.100e.", "cmac.no.reset.100e.")

nbr.agents <- 50 # 20 more than in Singh&Sutton (30)
nbr.episodes <- 100 # More than Singh&Sutton (20)
xints <- 8
vints <- 8
#alphas <- c(0.07,0.1,0.12,0.14,0.16,0.18,0.2,0.22)
lambdas <- c(0, 0.4, 0.7, 0.8, 0.9, 0.95, 0.99)
alphas <- c(0.06)
#lambdas <- c(0.9)
# Calculate all results.
# Observed with accumulating trace:
# From alpha >= 0.18 lambda 0.7 diverges (with regular CMAC, with some random CMACs it could go up to 0.2).
# From alpha >= 0.14 lambda 0.8 diverges (both kinds of CMAC).
# From alpha >= 0.1 lambda 0.9 gets stuck in local minimum, unable to get up from the hole. Then diverges on next episode. (with regular CMAC, with some random CMACs it could go up to 0.12.
# From alpha >= 0.06 lambda 0.95 diverges (with regular CMAC, with some random CMACs it could go up to 0.12).
# From alpha >= 0.02 lambda 0.99 diverges (with regular CMAC, with some random CMACs it could go up to 0.12).
# Observed with reset trace:
# From alpha >= 0.26, lambda 0.99 diverges (regular CMAC)
# Action-value function becomes completely erroneous.
#calculate.all(names, nbr.agents, nbr.episodes, classifier, alphas, lambdas)

# Or load them. But this also loads all old versions of functions,
# which can give bugs!
#load("D:/KF/Software/R/KFnnet/MC_CMAC_OIV.RData") 

# First plot: all alphas, lambdas
# alphas that worked for no.reset, lambdas 0.9 and 0.95
#alphas <- c(0.01,0.03,0.05,0.07,0.09,0.1,0.12,0.14,0.16,0.18,0.2,0.22,0.24,0.26)
# alphas common to both, lambdas 0.9 and 0.95
#alphas <- c(0.01,0.03,0.05,0.07,0.09,0.1,0.12,0.14,0.16)
# alphas common to all
#alphas <- c(0.01,0.05,0.07,0.1,0.14,0.16)
#alphas <- c(0.01,0.03,0.05,0.07,0.1,0.12,0.14,0.16,0.18,0.2,0.22)
alphas <- c(0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2,0.22,0.24,0.26,0.28,0.3)
lambdas <- c(0,0.4,0.7,0.8,0.9,0.95,0.99)
#plot.lines(names, alphas, lambdas, yrange=c(70,200), leftlabel="Steps/Episode. Averaged over first 100 episodes and 50 runs", experiment.nbr=3)

# Second plot: best results, with standard error (random offset CMAC)
best.alphas.acc <- c(0.22,0.22,0.12,0.12,0.1)
best.alphas.reset <- c(0.22,0.18,0.18,0.18,0.12,0.12,0.12)
best.alphas.no.reset <- c(0.22,0.2,0.16,0.16,0.2,0.2,0.12)
# Second plot: best results, with standard error (regular offset CMAC)
best.alphas.acc <- c(0.26,0.2,0.14,0.1,0.08,0.04)
best.alphas.reset <- c(0.28,0.24,0.28,0.3,0.24,0.22,0.12)
best.alphas.no.reset <- c(0.26,0.26,0.28,0.26,0.2,0.22,0.14)
best.alphas <- list(a=best.alphas.acc,b=best.alphas.reset,c=best.alphas.no.reset)
best.lambdas <- lambdas
#plot.final.comparison(names, lambdas, best.alphas, leftlabel=expression(paste("Average Steps/Episode at best ", alpha)), experiment.nbr=3, main.title="Mountain-Car, CMAC", min.y=80, max.y=120)
#a <- matrix(c(best.alphas.reset,best.alphas.no.reset),nrow=2,byrow=T)
#print.last.averages(names, lambdas, a, 10)

#======================================================================
# Next section contains calls for doing experiments in
# Singh&Sutton with 8x8 CMAC, discounted reward and
# zero step reward, 1 goal reward.
#
names <- c("disc.cmac.acc.100e.", "disc.cmac.reset.100e.", "disc.cmac.no.reset.100e.")

nbr.agents <- 50 # 20 more than in Singh&Sutton (30)
nbr.episodes <- 100 # More than Singh&Sutton (20)
dr <- 0.9
#alphas <- c(0.01,0.02,0.03,0.04,0.05)
alphas <- c(0.03,0.04)
#lambdas <- c(0, 0.4, 0.7, 0.8, 0.9, 0.95, 0.99)
lambdas <- c(0.4)
# Calculate all results
#calculate.all(names, nbr.agents, nbr.episodes, classifier, alphas, lambdas, dr=dr,epsilon=0.1,step.reward=0,goal.reward=1)

# Or load them. But this also loads all old versions of functions,
# which can give bugs!
#load("D:/KF/Software/R/KFnnet/CMAC_discountedRuns.RData") 

# First plot: all alphas, lambdas
alphas <- c(0.01,0.02,0.03,0.04,0.05)
lambdas <- c(0,0.4,0.7,0.8,0.9,0.95,0.99)
#plot.lines(names, alphas, lambdas, yrange=c(300,2500), leftlabel="Steps/Episode. Averaged over first 100 episodes and 50 runs", experiment.nbr=4)

# Second plot: best results, with standard error
#best.alphas.reset <- c(0.3,0.3,0.3,0.3,0.3,0.3,0.3)
#best.alphas.no.reset <- c(0.3,0.3,0.3,0.3,0.3,0.3,0.3)
best.alphas.reset <- c(0.05,0.05,0.05,0.02,0.02,0.04,0.02)
best.alphas.no.reset <- c(0.04,0.05,0.04,0.03,0.03,0.02,0.02)
best.lambdas <- lambdas
#plot.final.comparison(names, lambdas, best.alphas.reset, best.alphas.no.reset, leftlabel=expression(paste("Average Steps/Episode at best ", alpha)), experiment.nbr=4)
a <- matrix(c(best.alphas.reset,best.alphas.no.reset),nrow=2,byrow=T)
#print.last.averages(names, lambdas, a, 10)

#======================================================================
# From here on we have RBF tests.  
#
classifier <- rbf.classifier.new(nbrInputs=2, nbrOutputs=0, activation.function=squared.distance.activation, output.function=gaussian.output.function)
mins <- c(-1.2,-0.07)
maxs <- c(0.5,0.07)
at <- scale.translate.ranges(mins, maxs, c(0,0), c(1,1))
classifier$init.centroids.grid(mins, maxs, c(xints,vints),affine.transformation=at)
classifier$set.spread(0.01)
classifier$set.normalize(TRUE)
  
#======================================================================
# Next section contains calls for RBF experiments in
# Singh&Sutton. 
#
names <- c("rbf.acc.100e.", "rbf.reset.100e.", "rbf.no.reset.100e.")

nbr.agents <- 50 # 20 more than in Singh&Sutton (30)
nbr.episodes <- 100 # More than Singh&Sutton (20)
xints <- 8
vints <- 8
#alphas <- c(0.07,0.1,0.12,0.14,0.16,0.18,0.2,0.22)
lambdas <- c(0, 0.4, 0.7, 0.8, 0.9, 0.95, 0.99)
alphas <- c(3.1,3.3)
#lambdas <- c(0.9)

# Calculate all results.
# Observed with accumulating trace:
# From alpha >= 0.3 lambda 0.99 diverges.
# From alpha >= 0.9 lambda 0.95 diverges.
# From alpha >= 1.3 lambda 0.9 diverges.
# From alpha >= 2.3 lambda 0.8 diverges.
# Observed with replacing trace:
# From alpha >= 3.7 lambda 0.95, 0.99 diverges.
# From alpha >= 5.1 everything diverges (including lambda=0)
#calculate.all(names, nbr.agents, nbr.episodes, classifier, alphas, lambdas)

# Or load them. But this also loads all old versions of functions,
# which can give bugs!
#load("D:/KF/Software/R/KFnnet/MC_RBF_OIV.RData") 

# First plot: all alphas, lambdas
# alphas that worked for no.reset, lambdas 0.9 and 0.95
#alphas <- c(0.01,0.03,0.05,0.07,0.09,0.1,0.12,0.14,0.16,0.18,0.2,0.22,0.24,0.26)
# alphas common to both, lambdas 0.9 and 0.95
#alphas <- c(0.01,0.03,0.05,0.07,0.09,0.1,0.12,0.14,0.16)
# alphas common to all
#alphas <- c(0.01,0.05,0.07,0.1,0.14,0.16)
#alphas <- c(0.01,0.03,0.05,0.07,0.1,0.12,0.14,0.16,0.18,0.2,0.22)
alphas <- c(0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.9,2.1,2.3,2.5,2.7,2.9,3.1,3.3,3.5,3.7,3.9,4.1,4.3,4.5,4.7,4.9)
lambdas <- c(0,0.4,0.7,0.8,0.9,0.95,0.99)
#plot.lines(names, alphas, lambdas, yrange=c(70,200), leftlabel="Steps/Episode. Averaged over first 100 episodes and 50 runs", experiment.nbr=5)

# Second plot: best results, with standard error 
best.alphas.acc <- c(4.7,3.3,2.3,2.1,0.9,0.5,0.1)
best.alphas.reset <- c(0.1,0.1,0.1,0.1,0.1,0.1,0.1)
best.alphas.no.reset <- c(4.9,4.3,4.3,4.7,3.9,3.3,2.5)
best.alphas <- list(a=best.alphas.acc,b=best.alphas.reset,c=best.alphas.no.reset)
best.lambdas <- lambdas
#plot.final.comparison(names, lambdas, best.alphas, leftlabel=expression(paste("Average Steps/Episode for 50 runsand 100 episodes at best ", alpha)), experiment.nbr=5, main.title="Mountain-Car, RBF", min.y=70, max.y=110)
#a <- matrix(c(best.alphas.reset,best.alphas.no.reset),nrow=2,byrow=T)
#print.last.averages(names, lambdas, a, 10)

#======================================================================
# Next section contains calls for producing the plot that compares 
# Singh&Sutton's results with CMAC/RBF for ICINCO 2008 paper. 
#
names <- c("cmac.acc.100e.", "cmac.reset.100e.","rbf.acc.100e.", "rbf.no.reset.100e.")
cmac.alphas.acc <- c(0.26,0.2,0.14,0.1,0.08,0.04)
cmac.alphas.reset <- c(0.28,0.24,0.28,0.3,0.24,0.22,0.12)
cmac.alphas.no.reset <- c(0.26,0.26,0.28,0.26,0.2,0.22,0.14)
rbf.alphas.acc <- c(4.7,3.3,2.3,2.1,0.9,0.5,0.1)
rbf.alphas.reset <- c(0.1,0.1,0.1,0.1,0.1,0.1,0.1)
rbf.alphas.no.reset <- c(4.9,4.3,4.3,4.7,3.9,3.3,2.5)
best.alphas <- list(a=cmac.alphas.acc,b=cmac.alphas.no.reset,c=rbf.alphas.acc,d=rbf.alphas.no.reset)
best.lambdas <- lambdas

# Get plot with same number of agents episodes as in Singh&Sutton
#plot.final.comparison(names, lambdas, best.alphas, leftlabel=expression(paste("Average Steps/Episode at best ", alpha)), experiment.nbr=6, main.title="Mountain-Car, CMAC and RBF", min.y=100, max.y=270, nbr.plot.agents=30, nbr.plot.episodes=20)

#======================================================================
# Function for producing the ICINCO 2008 graph that allows to compare
# with results in Doya, 2000 and Schaal, 1997
#
mc.plot.icinco08 <- function() {
  # Always create new device so that we get the size we want
  x11(width=5,height=5)

  # We take the parameters that produced the best results.
  results <- rbf.no.reset.100e.a3.9_l0.9

  # Plot the graph
  plot(colMeans(results), type="l", xlab="Episode", ylab="Average Steps/Episode") 
}
