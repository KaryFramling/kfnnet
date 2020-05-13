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
  persp(x, y, zm, xlab="x", ylab="y", zlab="z", theta = -35, phi = 15, ticktype = "detailed")
}

# Example non-linear function
Fig.nonlinear.model <- function(xp=0.1,yp=0.2) {
  nlmodel <- nonlineardssmodel.two.inputs.new()
  z <- nlmodel$eval(pm)
  zm <- matrix(z, nrow = length(x), byrow = TRUE)
  vt <- persp(x, y, zm, xlab = "x", ylab = "y", zlab = "z", theta = -35, phi = 15, ticktype = "detailed") # persp3D might want these: , bg="white", colvar=NULL, col="black", facets=FALSE
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
  hist3D(x,y,z=zm, xlab="x", ylab="y", zlab="z", xlim=c(0,1), ylim=c(0,1), zlim=c(0,1), ticktype = "detailed", theta = -35, phi = 15, bg="white", colvar=NULL, col="black", facets=FALSE)
}

# Non-linear function, as function of x and of y
Fig.nonlinear.for.x.and.y <- function(xp=0.1, yp=0.2) {
  nlmodel <- nonlineardssmodel.two.inputs.new()
  xm <- matrix(x, nrow=length(x), ncol=2)
  xm[,2] <- yp
  z1 <- nlmodel$eval(xm)
  # Plot Figures side by side
  def.par <- par(no.readonly = TRUE) # save default, for resetting...
  layout(matrix(c(1,2), 1, 2, byrow = TRUE))
  # First figure
  plot(x, z1, type='l', xlab="x (with constant y=0.2)", ylab="z", xlim=c(0,1), ylim=c(0,1))
  outp <- nlmodel$eval(matrix(c(xp,yp),nrow=1))
  points(x=xp, y=outp, col = "red", pch = 16, cex = 2)
  text(x=c(0,0), y=c(0,1),c("absmin = 0.0","absmax = 1.0"),col="blue", pos=4)
  cmin <- nlmodel$eval(matrix(c(0,yp), nrow=1))
  cmax <- nlmodel$eval(matrix(c(1,yp), nrow=1))
  text(x=c(1,1), y=c(cmin,cmax),c(paste("Cmin = ", cmin), paste("Cmax = ", cmax)),col="blue", pos=2)
  text(x=xp+0.1, y=outp,c(paste("out =", outp)),col="blue",pos=4)
  arrows(x0=xp+0.1,y0=outp,x1=xp+0.01,col="red")
  # Then next one
  xm <- matrix(x, nrow=length(x), ncol=2)
  xm[,1] <- xp
  z2 <- nlmodel$eval(xm)
  plot(x, z2, type='l', xlab="y (with constant x=0.1)", ylab="z", xlim=c(0,1), ylim=c(0,1))
  points(x=yp, y=outp, col = "red", pch = 16, cex = 2)
  text(x=c(0,0), y=c(0,1),c("absmin = 0.0","absmax = 1.0"),col="blue", pos=4)
  cmin <- nlmodel$eval(matrix(c(xp,0), nrow=1))
  cmax <- nlmodel$eval(matrix(c(xp,1), nrow=1))
  text(x=c(1,1), y=c(cmin,cmax),c(paste("Cmin = ", cmin), paste("Cmax = ", cmax)),col="blue", pos=2)
  text(x=yp, y=outp+0.1,c(paste("out =", outp)),col="blue",pos=3)
  arrows(x0=yp,y0=outp+0.1,y1=outp+0.01,col="red")
  par(def.par)  #- reset to default
}

# Classifier with two inputs, two classes.
Fig.two.class.example <- function() {
  def.par <- par(no.readonly = TRUE) # save current settings, for resetting...

  x2c <- y2c <- seq(0, 5, 0.25)
  l2c <- list(x2c,y2c)
  pm2c <- create.permutation.matrix(l2c)
  
  
  rbf <- rbf.new(2, 2, 0, activation.function = squared.distance.activation, output.function = imqe.output.function)
#  rbf$set.nrbf(TRUE)
  # rbf$set.spread(0.01)
  # train.in <- matrix(c(0.25, 0.25, 0.75, 0.75), ncol=2, byrow=TRUE)
  rbf$set.spread(0.1)
  train.in <- matrix(c(1, 1, 4, 4), ncol=2, byrow=TRUE)
  train.target <- matrix(c(1,0,0,1), ncol=2, byrow=TRUE)
  train.inka(rbf, train.in, train.target, c=0.1, max.iter=2, classification.error.limit=0)
#  z <- rbf$eval(pm)
  z2c <- rbf$eval(pm2c)
  #zmax <- apply(z2c, 1, max)
  #zplot <- matrix(zmax, nrow=length(x2c), ncol=length(y2c))
  library(rgl) # See e.g. https://r.789695.n4.nabble.com/two-perspective-plots-in-in-plot-td818125.html
  persp3d(x2c,y2c,z2c[,1], col="green", aspect="iso", axes=T, box=T, xlab="X", ylab="Y", zlab="Z")
  persp3d(x2c,y2c,z2c[,2], col="red", add=TRUE)
  # fcol <- matrix("green3", nrow=nrow(zmax))
  # fcol[,z2c[,1]>z2c[,2]] <- "red"
  # fill <- matrix("green3", nrow = nrow(zplot)-1, ncol = ncol(zplot)-1)
  # persp(x2c,y2c,zplot,theta = 30, phi = 30, scale=FALSE,shade=0.4,border=NA,box=FALSE, col=fill)

  # layout(matrix(c(1,2), 1, 2, byrow = TRUE))
  # CI.CU <- rbf.classification.test(indices=c(1), visualize.output.index=1)
  # print(CI.CU)
  # CI.CU <- rbf.classification.test(indices=c(2), visualize.output.index=2)
  # print(CI.CU)
  par(def.par)  #- reset to what it was before
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
  
  # 3D plots for Sepal Size vs Iris class and Petal size vs Iris class
  # Create CIU plots for all inputs separately
  def.par <- par(no.readonly = TRUE) # save default, for resetting...
#  layout(matrix(seq(1:2), 1, 2, byrow = TRUE)) # Could probably use "par(mfrow, mfcol)" or split.screen also.
  par(mfrow=c(1,2))
  #par(mar = c(2,2,1,0)) # c(bottom, left, top, right)
  for ( out.ind in 1:length(iris.types) ) {
    inp.indices <- c(1,2)
    ciu$plot.CI.CU.3D(instance.values, ind.inputs=inp.indices, ind.output=out.ind, n.points=20,
                      theta = 0, phi = 15)
    inp.indices <- c(3,4)
    ciu$plot.CI.CU.3D(instance.values, ind.inputs=inp.indices, ind.output=out.ind, n.points=20,
                      theta = 0, phi = 15)
  }
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

Iris.Lime.Inka <- function() {
  predict_model.FunctionApproximator <- function(x, newdata, type, ...) {
    as.data.frame(x$eval(as.matrix(newdata)))
  }
  model_type.FunctionApproximator <- function(x, ...) "classification"
  explainer <- lime(iris[,1:4], iris.model.inka)
  iris_test <- iris[1,1:4]
  explanation <- explain(iris_test, explainer, n_labels = 3, n_features = 4)
  print(explanation)
  plot_features(explanation)
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
  par(mfrow=c(1,2))
  for ( ind.output in 1:length(iris.types) ) {
    ciu$barplot.CI.CU(instance.values, ind.inputs=c(1,2), ind.output=ind.output, neutral.CU=0.5, montecarlo.samples=1000,
                      target.concept="Sepal size and shape")
    ciu$barplot.CI.CU(instance.values, ind.inputs=c(3,4), ind.output=ind.output, neutral.CU=0.5, montecarlo.samples=1000,
                      target.concept="Petal size and shape")
  }
  
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
  par(mfrow=c(1,3))
#  par(mar = c(1,1,1,1)) # c(bottom, left, top, right)
  ciu.gbm.fa$barplot.CI.CU(Boston[406,1:n.in], main="Row #406, medv=5k$")
  ciu.gbm.fa$barplot.CI.CU(Boston[6,1:n.in], main="Row #6, medv=28.7k$")
  ciu.gbm.fa$barplot.CI.CU(Boston[370,1:n.in], main="Row #370, medv=50k$")
  par(mfrow=c(1,1))
  par(def.par)
}
