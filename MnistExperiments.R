# Tests and utilities for MNIST data set. 
#
# Created by Kary Främling 4oct2019
#

source("MnistUtilities.R")
source("Functions.R")
source("Interfaces.R")
source("NeuralLayer.R")
source("Adaline.R")
source("RBF.R")

Training <- load_label_file("train-labels-idx1-ubyte")
Test <- load_label_file("t10k-labels-idx1-ubyte")
matriximage <- load_image_file("train-images-idx3-ubyte")
matrixtest <- load_image_file("t10k-images-idx3-ubyte")

# Create target value vectors. There must exist more efficient way to do this also but this works...
t0 <- as.numeric(Training==0)
t1 <- as.numeric(Training==1)
t2 <- as.numeric(Training==2)
t3 <- as.numeric(Training==3)
t4 <- as.numeric(Training==4)
t5 <- as.numeric(Training==5)
t6 <- as.numeric(Training==6)
t7 <- as.numeric(Training==7)
t8 <- as.numeric(Training==8)
t9 <- as.numeric(Training==9)
t <- cbind(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9)

# Ideas for INKA
# 1. Current "uninformed" policy (learning for all outputs at same time)
# 2. Separate networks for every output
# 3. Initialize hidden layer with centroid RBF for every class

# Test with one class at a time, separate networks.
t.in <- matriximage$x
n.in <- ncol(t.in)
#targets <- cbind(t0,t1,t2,t3,t4,t5,t6,t7,t8,t9)
targets <- cbind(t0,t1)
n.out <- ncol(targets)
in.mins <- apply(t.in, 2, min)
in.maxs <- apply(t.in, 2, max) # This is not 255 for all pixels in the data! It's even zero for many of them. 
rbf <-
  rbf.new(n.in,
          n.out,
          0,
          activation.function = squared.distance.activation,
          output.function = imqe.output.function)
rbf$set.nrbf(TRUE)
# Maybe no need to normalize inputs here? 
#aff.trans <- scale.translate.ranges(in.mins, in.maxs, c(0,0,0,0), c(1,1,1,1))
ol <- rbf$get.outlayer()
ol$set.use.bias(FALSE)
rbf$set.spread(0.1) # d^2 parameter in INKA
c <- 1 # The "c" parameter in INKA training, minimal distance for adding new hidden neuron.
n.hidden <-
  train.inka(
    rbf,
    t.in,
    targets,
    c,
    max.iter = 200,
    inv.whole.set.at.end = F,
    classification.error.limit = 0
  )
# Calculate error measure etc.
y <- rbf$eval(t.in)
classification <- (y == apply(y, 1, max)) * 1
nbr.errors <-
  sum(abs(targets - classification)) / 2 # How many are mis-classified. This calculation is only correct if all classes t0-t9 are included!
cat("nbr.errors = ", nbr.errors, "\n")
cat("Number of hidden neurons: ", n.hidden, "\n")

