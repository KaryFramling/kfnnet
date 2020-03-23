source("Functions.R")
source("Adaline.R")
source("SarsaLearner.R")
source("BIMM.R")
source("EligibilityTrace.R")

run.grid <- function(plot=TRUE, save=FALSE) {
  source("GridTask.R")
  w <- 10
  h <- 10
  nagents <- 10
  neps <- 100
  random.transition.rate <- 0.2
  
  # Sarsa
  cnt.sarsa <- sarsa.grid(w, h, nagents,neps,lr=0.1,dr=0.9,lambda=0.9,epsilon=0.1, random.transition.rate, step.reward=0.0, goal.reward=1.0)
  if (plot)
    plot(1:neps,colMeans(cnt.sarsa),type='l')
  if ( save ) 
    write.table(cnt.sarsa,"GridRes/Sarsa_20x20_lr01_dr09_lmda09_e01.txt", row.names=F,col.names=F)

  # Sarsa OIV
  cnt.oiv <- sarsa.grid(w, h, nagents,neps,lr=1.0, dr=1.0, lambda=0.9, epsilon=0.0, random.transition.rate, step.reward=-1, goal.reward=0)
  if (plot)
    lines(1:neps,colMeans(cnt.oiv),type='l', col='green')
  if ( save ) 
    write.table(cnt.oiv,"GridRes/OIV_20x20_lr1_dr1_lmda09_e0.txt", row.names=F,col.names=F)
  
  # BIMM
  #cnt.bimm <-bimm.lw(nstates,nagents,neps,k.stm=0.000001,k.ltm=1.0,slap.lr=1,sarsa.lr=0.1,dr=0.9,lambda=0.9,epsilon = 0.0)
  #if (plot)
    #lines(1:neps,colMeans(cnt.bimm),type='l', col='red')
  #if ( save ) 
    #write.table(cnt.bimm,"GridRes/BIMM_LW10_stmw0000001_ltmw1_slr1_qlr01_dr09_lmda09_e0.txt", row.names=F,col.names=F)
    
  list (
        get.cnt.sarsa = function() { cnt.sarsa },
        get.cnt.oiv = function() { cnt.oiv },
        get.cnt.bimm = function() { cnt.bimm },
        )
}

#source("MountainCar.R")
#mcdata <- run.mountaincar(save=FALSE, plot=TRUE)
#source("LinearWalk.R")
#lwdata <- run.linearwalk(save=FALSE, plot=TRUE)
#griddata <- run.grid(save=FALSE, plot=TRUE)

