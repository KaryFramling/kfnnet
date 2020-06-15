# Test scripts for INKA network.
#
# Kary Främling, 21 Apr 2020
#

library(caret) # Includes some nice utility functions.


source("RBF.R")

# For reproducibility. "set.seed(2)" gives zero test set error with three hidden units.
set.seed(2)

# Iris data set apparently exists by default in R? Split into training and test set
inTrain <- createDataPartition(y=iris$Species, p=0.75, list=FALSE) # 75% to train set
training.Iris <- iris[inTrain,]
testing.Iris <- iris[-inTrain,]
rbf <- train.inka.formula(Species~., data=training.Iris, spread=0.1, max.iter=20, classification.error.limit=0,)
print(paste("Number of hidden neurons:", nrow(rbf$get.hidden()$get.weights())))
y <- rbf$eval(as.matrix(testing.Iris[,1:4]))
classification <- (y == apply(y, 1, max)) * 1;
#perf <- sum(abs(test.out - classification)) / 2; print(perf) # Simple calculation of how many mis-classified
# Confusion matrix. Requires converting one-hot to factor.
pred.class <- seq(1:nrow(classification)); for ( i in 1:3) { pred.class[classification[,i]==1] <- levels(iris$Species)[i]}
confusionMatrix(data = as.factor(pred.class), reference = iris[-inTrain, 5])

# Then use the "old version" (no formula) for finding best result. 
targets <- model.matrix(~0+iris[,'Species']) # One-hot encoding
n.out <- ncol(targets)
t.in <- as.matrix(iris[inTrain, 1:4]) 
t.out <- targets[inTrain,]
test.in <- as.matrix(iris[-inTrain, 1:4])
test.out <- targets[-inTrain,]
n.rbfs.to.test <- 10 # Let's try with 10 networks and use the best one. 
# Utility function for finding the best among "n" trained networks.
rbf <- find.best.inka(n=n.rbfs.to.test, train.inputs=t.in, train.outputs=t.out, max.iter=30, 
                      inv.whole.set.at.end=F, classification.error.limit=0, 
                      rmse.limit=NULL, activation.function=squared.distance.activation, 
                      output.function=imqe.output.function, nrbf=T, use.bias=F, 
                      spread=0.1, c=0.01) #, test.inputs=test.in, test.outputs=test.out
out <- rbf$eval(test.in[1,]); print(out) # Setosa so first output should be highest
out <- rbf$eval(test.in[16,]); print(out) # Versicolor so second output should be highest
out <- rbf$eval(test.in[31,]); print(out) # Virginica so third output should be highest
nrow(rbf$get.hidden()$get.weights()) # Number of rows is number of hidden neurons

# See how well we performed
y <- rbf$eval(test.in)
classification <- (y == apply(y, 1, max)) * 1; 
perf <- sum(abs(test.out - classification)) / 2; print(perf) # Simple calculation of how many mis-classified

# Confusion matrix. Requires converting one-hot to factor. 
pred.class <- seq(1:nrow(classification)); for ( i in 1:3) { pred.class[classification[,i]==1] <- levels(iris$Species)[i]}
confusionMatrix(data = as.factor(pred.class), reference = iris[-inTrain, 5])

