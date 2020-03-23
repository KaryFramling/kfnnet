# Scripts for producing EXTTRAAMAS 2020 article figures and plots.

source("Functions.R")
source("ContextualImportanceUtility.R")
source("IrisExperiments.R")

library(plot3D)

# Create input matrix for evaluation&plots
x <- y <- seq(0, 1, 0.05)
l <- list(x,y)
pm <- create.permutation.matrix(l)

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

Fig.fuzzy.AND <- function() {
  def.par <- par(no.readonly = TRUE) # save current settings, for resetting...
  layout(matrix(c(1,2), 1, 2, byrow = TRUE))
  CI.CU <- rbf.classification.test(indices=c(1), visualize.output.index=1)
  print(CI.CU)
  CI.CU <- rbf.classification.test(indices=c(2), visualize.output.index=2)
  print(CI.CU)
  par(def.par)  #- reset to what it was before
}
               
Fig.iris.plots <- function() {
  # Get INKA network trained on Iris data
  rbf <- iris.inka.test()
  
  # We need to re-create these for plotting etc.
  t.in <-
    as.matrix(iris[, 1:4]) # Iris data set apparently exists by default in R
  in.mins <- apply(t.in, 2, min)
  in.maxs <- apply(t.in, 2, max)
  
  # Labels
  iris.inputs <- c("Sepal Length", "Sepal width", "Petal Length", "Petal width") # In centimeters
  iris.types <- c("Setosa", "Versicolor", "Virginica")
  
  # Plot effect of one input on one output for given instance.
  instance.values <- c(7, 3.2, 6, 1.8) # This is not a flower from Iris set!
  print("Values of the three outputs.")
  print(rbf$eval(instance.values))

  # Plot Figures side by side
  def.par <- par(no.readonly = TRUE) # save default, for resetting...
  layout(matrix(seq(1:4), 2, 2, byrow = TRUE)) # Could probably use "par(mfrow, mfcol)" or split.screen also.

  # Create CIU plots for all inputs separately
  par(mar = c(5,5,1,1)) # c(bottom, left, top, right)
  for ( iris.ind in 1:length(iris.types) ) {
    for ( inp.ind in 1:length(iris.inputs) ) {
      plot.CI.CU(rbf, instance.values, inp.ind, iris.ind, in.mins, in.maxs, xlab=iris.inputs[inp.ind], ylab=iris.types[iris.ind], ylim=c(0,1)) # No effect with "mar=c(0,0,0,0)"?
    }
  }
  par(def.par)  #- reset to what it was before
    
  # CI&CU values for every input 
  c.minmax <- cbind(in.mins, in.maxs)
  for ( inp.ind in 1:length(iris.inputs) ) {
    CI.CU <- contextual.IU(rbf, instance.values, c(inp.ind),matrix(c(0,1,0,1,0,1), ncol = 2, byrow = T), montecarlo.samples = 1000, c.minmax = c.minmax)
    print(iris.inputs[inp.ind])
    print(CI.CU)
  }

  # CI&CU values for Sepal size 
  CI.CU <- contextual.IU(rbf, instance.values, c(1,2),matrix(c(0,1,0,1,0,1), ncol = 2, byrow = T), montecarlo.samples = 1000, c.minmax = c.minmax)
  print("Sepal size")
    print(CI.CU)
  
  # CI&CU values for Petal size 
  CI.CU <- contextual.IU(rbf, instance.values, c(3,4),matrix(c(0,1,0,1,0,1), ncol = 2, byrow = T), montecarlo.samples = 1000, c.minmax = c.minmax)
  print("Petal size")
  print(CI.CU)
  
  # CI&CU values for all inputs
  CI.CU <- contextual.IU(rbf, instance.values, c(1,2,3,4),matrix(c(0,1,0,1,0,1), ncol = 2, byrow = T), montecarlo.samples = 1000, c.minmax = c.minmax)
  print("All inputs")
  print(CI.CU)
  
  # 3D plots for Sepal Size vs Iris class and Petal size vs Iris class
  # Create CIU plots for all inputs separately
  #par(mar = c(5,5,1,1)) # c(bottom, left, top, right)
  layout(matrix(seq(1:2), 1, 2, byrow = TRUE)) # Could probably use "par(mfrow, mfcol)" or split.screen also.
  #par(mar = c(2,2,1,0)) # c(bottom, left, top, right)
  for ( out.ind in 1:length(iris.types) ) {
    inp.indices <- c(1,2)
    plot.CI.CU.3D(rbf, instance.values, inp.indices, out.ind, in.mins, in.maxs, n.points=20, 
                  xlab=iris.inputs[inp.indices[1]], ylab=iris.inputs[inp.indices[2]], zlab=iris.types[out.ind], 
                  theta = 0, phi = 15, zlim=c(0,1))
    inp.indices <- c(3,4)
    plot.CI.CU.3D(rbf, instance.values, inp.indices, out.ind, in.mins, in.maxs, n.points=20, 
                  xlab=iris.inputs[inp.indices[1]], ylab=iris.inputs[inp.indices[2]], zlab=iris.types[out.ind], 
                  theta = 0, phi = 15, zlim=c(0,1))
  }
  par(def.par)  #- reset to what it was before
  
  # layout(matrix(c(1)))
}