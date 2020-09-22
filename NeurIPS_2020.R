# Scripts for producing NeurIPS 2020 article figures and plots.

source("Functions.R")
source("RBF.R")
source("ContextualImportanceUtility.R")
source("IrisExperiments.R")

library(MASS)
library(plot3D)
library(lime)
library(caret)

# Create input matrix for evaluation&plots
x <- y <- seq(0, 1, 0.05)
l <- list(x,y)
pm <- create.permutation.matrix(l)

# Create all models used here in this script
iris_train <- iris[, 1:4]
iris_lab <- iris[[5]]
iris.model.lda <- lda(iris_train, iris_lab)
iris.model.rf <- train(iris_train, iris_lab, method = 'rf')
iris.model.inka <- iris.get.best.inka()


# Simple weighted sum
Fig.weighted.sum <- function() {
  ws <- weighted.sum.new(c(0.3, 0.7))
  z <- ws$eval(pm)
  zm <- matrix(z, nrow = length(x), byrow = TRUE)
  persp(x, y, zm, xlab="x1", ylab="x2", zlab="y", theta = -35, phi = 15, ticktype = "detailed")
}

# Example non-linear function
Fig.nonlinear.model <- function(xp=0.1,yp=0.2) {
  nlmodel <- nonlineardssmodel.two.inputs.new()
  z <- nlmodel$eval(pm)
  zm <- matrix(z, nrow = length(x), byrow = TRUE)
  vt <- persp(x, y, zm, xlab = "x1", ylab = "x2", zlab = "y", theta = -35, phi = 15, ticktype = "detailed") # persp3D might want these: , bg="white", colvar=NULL, col="black", facets=FALSE
  points(trans3d(xp, yp, nlmodel$eval(matrix(c(xp,yp),nrow=1)), pmat = vt), col = "red", pch = 16, cex = 3)
}

# Example rule-based histogram
Fig.rule.histogram <- function() {
  z <- matrix(0.0, nrow=nrow(pm)) # Default value
  z[pm[,1]>0.1] <- 0.2
  z[pm[,1]>0.3 & pm[,2]>0.1] <- 0.6
  z[pm[,1]>0.5 & pm[,2]>0.25 & pm[,2]<0.75] <- 1.0
  z[pm[,2]>0.8 & pm[,1]>0.25 & pm[,1]<0.5] <- 0.4
  #  z[pm[,1]>0.8 & pm[,2]>0.8 & pm[,2]<0.9] <- 0.7
  zm <- matrix(z,nrow=length(x))
  hist3D(x,y,z=zm, xlab="x1", ylab="x2", zlab="y", xlim=c(0,1), ylim=c(0,1), zlim=c(0,1), ticktype = "detailed", theta = -35, phi = 15, bg="white", colvar=NULL, col="black", facets=FALSE)
}

# Non-linear function, as function of x and of y
CIU.simple.nonlinear <- function(xp=0.1, yp=0.2) {
  nlmodel <- nonlineardssmodel.two.inputs.new()
  xm <- matrix(x, nrow=length(x), ncol=2)
  xm[,2] <- yp
  z1 <- nlmodel$eval(xm)
  # Plot Figures side by side
  def.par <- par(no.readonly = TRUE) # save default, for resetting...
  layout(matrix(c(1,2), 1, 2, byrow = TRUE))
  # First figure
  plot(x, z1, type='l', xlab="x1 (with constant x2=0.2)", ylab="y", xlim=c(0,1), ylim=c(0,1), asp=1)
  outp <- nlmodel$eval(matrix(c(xp,yp),nrow=1))
  points(x=xp, y=outp, col = "red", pch = 16, cex = 2)
  cmin <- nlmodel$eval(matrix(c(0,yp), nrow=1))
  cmax <- nlmodel$eval(matrix(c(1,yp), nrow=1))
  abline(h=c(0,1,cmin,cmax),col="grey60")
  text(x=c(0,0), y=c(-0.025,1-0.025),c("absmin = 0.0","absmax = 1.0"),col="blue", pos=4)
  text(x=c(1,1), y=c(cmin+0.025,cmax+0.025),c(paste("Cmin = ", as.character(round(cmin, 3))), paste("Cmax = ", as.character(round(cmax, 3)))),col="blue", pos=2)
  text(x=xp+0.1, y=outp,c(paste("out =", as.character(round(outp,3)))),col="blue",pos=4)
  arrows(x0=xp+0.1,y0=outp,x1=xp+0.01,col="red")
  # Then next one
  xm <- matrix(x, nrow=length(x), ncol=2)
  xm[,1] <- xp
  z2 <- nlmodel$eval(xm)
  plot(x, z2, type='l', xlab="x2 (with constant x1=0.1)", ylab="y", xlim=c(0,1), ylim=c(0,1), asp=1)
  points(x=yp, y=outp, col = "red", pch = 16, cex = 2)
  cmin <- nlmodel$eval(matrix(c(xp,0), nrow=1))
  cmax <- nlmodel$eval(matrix(c(xp,1), nrow=1))
  abline(h=c(0,1,cmin,cmax),col="grey60")
  text(x=c(0,0), y=c(-0.025,1-0.025),c("absmin = 0.0","absmax = 1.0"),col="blue", pos=4)
  text(x=c(1,1), y=c(cmin+0.025,cmax+0.025),c(paste("Cmin = ", as.character(round(cmin, 3))), paste("Cmax = ", as.character(round(cmax, 3)))),col="blue", pos=2)
  text(x=yp, y=outp+0.1,c(paste("out =", as.character(round(outp,3)))),col="blue",pos=3)
  arrows(x0=yp,y0=outp+0.1,y1=outp+0.01,col="red")
  par(def.par)  #- reset to default
}

Fig.iris.plots.NeurIPS <- function() {

  # We need to re-create these for plotting etc.
  t.in <-
    as.matrix(iris[, 1:4])
  in.mins <- apply(t.in, 2, min)
  in.maxs <- apply(t.in, 2, max)
  c.minmax <- cbind(in.mins, in.maxs)
  
  # Labels
  iris.inputs <- c("Sepal Length", "Sepal width", "Petal Length", "Petal width") # In centimeters
  iris.types <- c("Setosa", "Versicolor", "Virginica")
  
  # Plot effect of one input on one output for given instance.
  # inka takes matrix, others use data.frame.
  use.inka <- TRUE
  print("Values of the three outputs.")
  if ( use.inka ) {
    instance.values <- c(7, 3.2, 6, 1.8) # This is not a flower from Iris set!
    model <- iris.model.inka
    print(model$eval(instance.values))
  }
  else {
    instance.values <- iris[1,1:4]
    instance.values[1,] <- c(7, 3.2, 6, 1.8)
    model <- iris.model.rf
    print(predict(model, instance.values, type="prob"))
  }
  
  # Initialize CIU object
  ciu <- ciu.new(model, in.min.max.limits=c.minmax, abs.min.max=matrix(c(0,1,0,1,0,1), ncol = 2, byrow = T), input.names=iris.inputs, output.names=iris.types)

  # Plot Figures side by side
  def.par <- par(no.readonly = TRUE) # save default, for resetting...
  par(mfrow=c(2,2))
  
  # Create CIU plots for all inputs separately
  par(mar = c(5,5,1,1)) # c(bottom, left, top, right)
  for ( iris.ind in 1:length(iris.types) ) {
    for ( inp.ind in 1:length(iris.inputs) ) {
      #      plot.CI.CU(rbf, instance.values, inp.ind, iris.ind, in.mins, in.maxs, xlab=iris.inputs[inp.ind], ylab=iris.types[iris.ind], ylim=c(0,1)) # No effect with "mar=c(0,0,0,0)"?
      ciu$plot.CI.CU(instance.values, ind.input=c(inp.ind), ind.output=c(iris.ind), 
                     n.points=40)
    }
  }
  par(def.par)
  
  # CI&CU values for every input 
  for ( inp.ind in 1:length(iris.inputs) ) {
    CI.CU <- ciu$explain(instance.values, ind.inputs.to.explain=c(inp.ind), montecarlo.samples = 1000)
    print(iris.inputs[inp.ind])
    print(CI.CU)
  }
  
  # CI&CU values for Sepal size 
  CI.CU <- ciu$explain(instance.values, ind.inputs.to.explain=c(1,2), montecarlo.samples = 1000)
  print("Sepal size")
  print(CI.CU)
  
  # CI&CU values for Petal size 
  CI.CU <- ciu$explain(instance.values, ind.inputs.to.explain=c(3,4), montecarlo.samples = 1000)
  print("Petal size")
  print(CI.CU)
  
  # CI&CU values for all inputs
  CI.CU <- ciu$explain(instance.values, ind.inputs.to.explain=c(1:4), montecarlo.samples = 1000)
  print("All inputs")
  print(CI.CU)
  
  # # 3D plots for Sepal Size vs Iris class and Petal size vs Iris class
  # def.par <- par(no.readonly = TRUE) # save default, for resetting...
  # par(mfrow=c(1,2))
  # #par(mar = c(2,2,1,0)) # c(bottom, left, top, right)
  # for ( out.ind in 1:length(iris.types) ) {
  #   inp.indices <- c(1,2)
  #   ciu$plot.CI.CU.3D(instance.values, ind.inputs=inp.indices, ind.output=out.ind, n.points=20,
  #                     theta = 0, phi = 15)
  #   inp.indices <- c(3,4)
  #   ciu$plot.CI.CU.3D(instance.values, ind.inputs=inp.indices, ind.output=out.ind, n.points=20,
  #                     theta = 0, phi = 15)
  # }
  # par(def.par)  #- reset to what it was before

  # 3D plots for Petal size vs Iris class
  def.par <- par(no.readonly = TRUE) # save default, for resetting...
  par(mfrow=c(1,2))
  inp.indices <- c(3,4)
  ciu$plot.CI.CU.3D(instance.values, ind.inputs=inp.indices, ind.output=2, n.points=20,
                      theta = 45, phi = 15)
  ciu$plot.CI.CU.3D(instance.values, ind.inputs=inp.indices, ind.output=3, n.points=20,
                    theta = 45, phi = 15)
  par(mfrow=c(1,1))
  par(def.par)  #- reset to what it was before
  
  # Barplots  
  par(mfrow=c(1,3))
  for ( out.ind in 1:length(iris.types) ) {
    ciu$barplot.CI.CU(inputs=instance.values, ind.output=out.ind)
  }
  par(mfrow=c(1,1))

  # Pie charts  
  #par(mfrow=c(1,3))
  for ( out.ind in 1:length(iris.types) ) {
    ciu$pie.CI.CU(inputs=instance.values, ind.output=out.ind)
  }
  par(mfrow=c(1,1))
}

# Tests with vocabulary, intermediate concepts. 
Iris.Intermediate.Concepts <- function() {
  # We need to re-create these for plotting etc.
  t.in <-
    as.matrix(iris[, 1:4])
  in.mins <- apply(t.in, 2, min)
  in.maxs <- apply(t.in, 2, max)
  c.minmax <- cbind(in.mins, in.maxs)
  
  # Labels
  iris.inputs <- c("Sepal Length", "Sepal width", "Petal Length", "Petal width") # In centimeters
  iris.types <- c("Setosa", "Versicolor", "Virginica")
  
  # Plot effect of one input on one output for given instance.
  instance.values <- c(7, 3.2, 6, 1.8) # This is not a flower from Iris set!
  print("Values of the three outputs.")
  print(iris.model.inka$eval(instance.values))

  # Small vocabulary
  voc <- list("Sepal size and shape"=c(1,2), "Petal size and shape"=c(3,4))

  # Initialize CIU object
  ciu <- ciu.new(iris.model.inka, in.min.max.limits=c.minmax, abs.min.max=matrix(c(0,1,0,1,0,1), ncol = 2, byrow = T), 
                 input.names=iris.inputs, output.names=iris.types, vocabulary=voc)
  CI.CU <- ciu$explain.vocabulary(instance.values, concepts.to.explain=c("Sepal size and shape","Petal size and shape"), 
                                  montecarlo.samples=1000)
  print(CI.CU)

  # Bar plots for intermediate concepts.
  par(mfrow=c(1,3))
  for ( ind.output in 1:length(iris.types) ) {
    ciu$barplot.CI.CU(instance.values, ind.output=ind.output, neutral.CU=0.5, montecarlo.samples=1000,
                    concepts.to.explain=c("Sepal size and shape","Petal size and shape"))
  }
  
  # CIU of input features versus intermediate concepts.
#  CI.CU <- ciu$explain(instance.values, ind.inputs.to.explain=c(1), montecarlo.samples=1000, 
#                      target.concept="Sepal size and shape")
  # par(mfrow=c(1,2))
  # for ( ind.output in 1:length(iris.types) ) {
  #   ciu$barplot.CI.CU(instance.values, ind.inputs=c(1,2), ind.output=ind.output, neutral.CU=0.5, montecarlo.samples=1000,
  #                     target.concept="Sepal size and shape")
  #   ciu$barplot.CI.CU(instance.values, ind.inputs=c(3,4), ind.output=ind.output, neutral.CU=0.5, montecarlo.samples=1000,
  #                     target.concept="Petal size and shape")
  # }
  par(mfrow=c(1,3))
  ciu$barplot.CI.CU(instance.values, ind.inputs=c(3,4), ind.output=1, neutral.CU=0.5, montecarlo.samples=1000,
                    target.concept="Petal size and shape")
  ciu$barplot.CI.CU(instance.values, ind.inputs=c(3,4), ind.output=2, neutral.CU=0.5, montecarlo.samples=1000,
                      target.concept="Petal size and shape")
  ciu$barplot.CI.CU(instance.values, ind.inputs=c(3,4), ind.output=3, neutral.CU=0.5, montecarlo.samples=1000,
                    target.concept="Petal size and shape")
  
  par(mfrow=c(1,1))
  ciu$pie.CI.CU(instance.values, ind.output=ind.output, montecarlo.samples=1000,
                    concepts.to.explain=c("Sepal size and shape","Petal size and shape"))
  ciu$pie.CI.CU(inputs=instance.values, ind.inputs=c(1,2), ind.output=ind.output, montecarlo.samples=1000,
                target.concept="Sepal size and shape")
  ciu$pie.CI.CU(inputs=instance.values, ind.inputs=c(3,4), ind.output=ind.output, montecarlo.samples=1000,
                target.concept="Petal size and shape")
  par(mfrow=c(1,1))
}

Boston.Figures.NeurIPS <- function() {
  require(MASS) # Just in case Boston is not already available
  require(gbm)
  
  n.in <- ncol(Boston) - 1
  in.mins <- apply(Boston[,1:n.in], 2, min)
  in.maxs <- apply(Boston[,1:n.in], 2, max)
  c.minmax <- cbind(in.mins, in.maxs)
  out.range <- matrix(c(min(Boston$medv), max(Boston$medv)), ncol=2)
  
  # We don't care about train/test set in this case because it's not about evaluating training performance.
  Boston.boost=gbm(medv ~ . ,data = Boston, distribution = "gaussian", n.trees=10000,
                   shrinkage = 0.01, interaction.depth = 4)
  gbm.fa.new <- function(gbm, n.trees=1) {
    o.gbm <- gbm
    o.n.trees <- n.trees
    pub <- list(eval = function(inputs) { predict(o.gbm,inputs,n.trees=o.n.trees) })
    class(pub) <- c("gbm.fa",class(function.approximator.new()),class(pub))
    return(pub)
  }
  gbm.fa <- gbm.fa.new(Boston.boost, 10000)
  ciu.gbm.fa <- ciu.new(gbm.fa, in.min.max.limits=c.minmax, abs.min.max=out.range, 
                        input.names=names(Boston)[1:n.in], output.names=names(Boston)[n.in+1])
  def.par <- par(no.readonly = TRUE) # save default, for resetting...

  # Instance 406 has medv=5
  # Instance 370 has medv=50
  # Instance 6 has medv=28.7
  # Instance 96 has medv=28.4
  # Boston$medv[362] = 19.9
  # Boston$medv[199] = 34.6
  par(mfrow=c(1,2))
  ciu.gbm.fa$barplot.CI.CU(Boston[406,1:n.in], main="Row #406, medv=5k$")
  ciu.gbm.fa$barplot.CI.CU(Boston[370,1:n.in], main="Row #370, medv=50k$")
  par(mfrow=c(1,1))
  par(def.par)
}

Boston.Figures.NeurIPS.Supplementary <- function() {
  require(MASS) # Just in case Boston is not already available
  require(gbm)
  
  n.in <- ncol(Boston) - 1
  in.mins <- apply(Boston[,1:n.in], 2, min)
  in.maxs <- apply(Boston[,1:n.in], 2, max)
  c.minmax <- cbind(in.mins, in.maxs)
  out.range <- matrix(c(min(Boston$medv), max(Boston$medv)), ncol=2)
  
  # We don't care about train/test set in this case because it's not about evaluating training performance.
  Boston.boost=gbm(medv ~ . ,data = Boston, distribution = "gaussian", n.trees=10000,
                   shrinkage = 0.01, interaction.depth = 4)
  gbm.fa.new <- function(gbm, n.trees=1) {
    o.gbm <- gbm
    o.n.trees <- n.trees
    pub <- list(eval = function(inputs) { predict(o.gbm,inputs,n.trees=o.n.trees) })
    class(pub) <- c("gbm.fa",class(function.approximator.new()),class(pub))
    return(pub)
  }
  gbm.fa <- gbm.fa.new(Boston.boost, 10000)
  ciu.gbm.fa <- ciu.new(gbm.fa, in.min.max.limits=c.minmax, abs.min.max=out.range, 
                        input.names=names(Boston)[1:n.in], output.names=names(Boston)[n.in+1])
  def.par <- par(no.readonly = TRUE) # save default, for resetting...
  
  # Instance 406 has medv=5
  # Instance 370 has medv=50
  # Instance 96 has medv=28.4
  # Instance 6 has medv=28.7
  # Boston$medv[362] = 19.9
  # Boston$medv[199] = 34.6
  bi <- c(406, 362, 6, 96, 199, 370)
  medv <- c(5, 19.9, 28.4, 28.7, 34.6, 50)
  par(mfrow=c(2,3))
  for ( i in 1:length(bi)) {
    ciu.gbm.fa$barplot.CI.CU(Boston[bi[i],1:n.in], main=paste("Row # ", bi[i], ", medv=", medv[i], "k$", sep=""))
  }
  
  # Create CIU plots for lstat
  par(mar = c(5,5,1,1)) # c(bottom, left, top, right)
  for ( i in 1:length(bi)) {
    ciu.gbm.fa$plot.CI.CU(Boston[bi[i],1:n.in], ind.input=13, ind.output=1, n.points=40, ylab="medv", 
                          main=paste("Row # ", bi[i], ", medv=", medv[i], "k$", sep=""))
  }

  # Create CIU plots for black
  par(mar = c(5,5,1,1)) # c(bottom, left, top, right)
  for ( i in 1:length(bi)) {
    ciu.gbm.fa$plot.CI.CU(Boston[bi[i],1:n.in], ind.input=12, ind.output=1, n.points=40, ylab="medv", 
                          main=paste("Row # ", bi[i], ", medv=", medv[i], "k$", sep=""))
  }
  
  # Create CIU plots for dis
  par(mar = c(5,5,1,1)) # c(bottom, left, top, right)
  for ( i in 1:length(bi)) {
    ciu.gbm.fa$plot.CI.CU(Boston[bi[i],1:n.in], ind.input=8, ind.output=1, n.points=40, ylab="medv", 
                          main=paste("Row # ", bi[i], ", medv=", medv[i], "k$", sep=""))
  }
  
  # Create CIU plots for rm
  par(mar = c(5,5,1,1)) # c(bottom, left, top, right)
  for ( i in 1:length(bi)) {
    ciu.gbm.fa$plot.CI.CU(Boston[bi[i],1:n.in], ind.input=6, ind.output=1, n.points=40, ylab="medv", 
                          main=paste("Row # ", bi[i], ", medv=", medv[i], "k$", sep=""))
  }
  
  # Create CIU plots for crim
  par(mar = c(5,5,1,1)) # c(bottom, left, top, right)
  for ( i in 1:length(bi)) {
    ciu.gbm.fa$plot.CI.CU(Boston[bi[i],1:n.in], ind.input=1, ind.output=1, n.points=40, ylab="medv", 
                          main=paste("Row # ", bi[i], ", medv=", medv[i], "k$", sep=""))
  }
  
  par(mfrow=c(1,1))
  par(def.par)
}
