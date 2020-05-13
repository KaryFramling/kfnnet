# Tests and utilities for Boston data set. 
#
# 13 input variables, continuous-valued output (last column, "medv"). 506 instances/rows. 
# Column explanations: 
# - CRIM     per capita crime rate by town
# - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
# - INDUS    proportion of non-retail business acres per town
# - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# - NOX      nitric oxides concentration (parts per 10 million)
# - RM       average number of rooms per dwelling
# - AGE      proportion of owner-occupied units built prior to 1940
# - DIS      weighted distances to five Boston employment centres
# - RAD      index of accessibility to radial highways
# - TAX      full-value property-tax rate per $10,000
# - PTRATIO  pupil-teacher ratio by town
# - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# - LSTAT    % lower status of the population
# - MEDV     Median value of owner-occupied homes in $1000's
#
# Created by Kary Främling 22apr2020
#

# Source about Gradient Boosting with Boston data set:
# https://datascienceplus.com/gradient-boosting-in-r/

# Boston is either included by default, or can be imported from MASS 
require(MASS)
require(caret) # Used here for data preparation functionalities. 

source("RBF.R")

# Prepare data
# caret preprocessing: 
# https://www.machinelearningplus.com/machine-learning/caret-package/#3datapreparationandpreprocessing
n.in <- ncol(Boston) - 1
n.out <- 1
preProcess_range_model <- preProcess(Boston, method='range') # Everything into [0,1]
t.in <- as.matrix(predict(preProcess_range_model, newdata = Boston)[,1:n.in]) # Normalised inputs
targets <- as.matrix(Boston$medv) # Original target values
trainRowNumbers <- createDataPartition(targets, p=0.8, list=FALSE)
train.in <- t.in[trainRowNumbers,]
test.in <- t.in[-trainRowNumbers,]
train.targets <- targets[trainRowNumbers,]
test.targets <- targets[-trainRowNumbers,]

# Create net and train
rbf <- rbf.new(n.in, n.out, 0, activation.function = squared.distance.activation,
               output.function = imqe.output.function)
rbf$set.nrbf(TRUE)
ol <- rbf$get.outlayer()
ol$set.use.bias(FALSE)
rbf$set.spread(0.1) # d^2 parameter in INKA
n.hidden <- train.inka(rbf, train.in, train.targets, c=0.01, max.iter=300, 
                       inv.whole.set.at.end=F, rmse.limit=2.5, test.inputs=test.in, test.outputs=test.targets)
print(paste("Hidden: ",n.hidden))
test.out <- rbf$eval(test.in)
#RMSE(test.targets,test.out) # How well did we do in practice?
root.mean.squared.error(test.out, test.targets)

# Same with Gradient Boosting
require(gbm)
Boston.boost=gbm(medv ~ . ,data = Boston[trainRowNumbers,],distribution = "gaussian",n.trees = 10000,
                 shrinkage = 0.01, interaction.depth = 4)
n.trees = seq(from=100 ,to=10000, by=100) # Don't know what this does in reality?
predmatrix<-predict(Boston.boost,Boston[-trainRowNumbers,],n.trees=n.trees)
#RMSE(as.matrix(Boston$medv)[-trainRowNumbers,],predmatrix)
root.mean.squared.error(predmatrix,as.matrix(Boston$medv)[-trainRowNumbers,])

# Try out a set of RBFs and take best one. Takes a little longer...
n.rbfs.to.test <- 10
rbf <- find.best.inka (n=n.rbfs.to.test, train.inputs=train.in, train.outputs=train.targets, max.iter=200, 
                       inv.whole.set.at.end=F, classification.error.limit=NULL, 
                       rmse.limit=2.5, activation.function=squared.distance.activation, 
                       output.function=imqe.output.function, nrbf=T, use.bias=F, 
                       spread=0.1, c=0.01, test.inputs=test.in, test.outputs=test.targets)
test.out <- rbf$eval(test.in)
root.mean.squared.error(test.out, test.targets)
print(paste("Hidden: ",nrow(rbf$get.hidden()$get.weights())))

# CIU
source("ContextualImportanceUtility.R")
in.mins <- apply(t.in, 2, min)
in.maxs <- apply(t.in, 2, max)
c.minmax <- cbind(in.mins, in.maxs)
out.range <- matrix(c(min(targets), max(targets)), ncol=2)
instance.values <- t.in[1,1:n.in] # Not very expensive
# instance.values <- t.in[370,1:n.in] # Very expensive
ciu.inka <- ciu.new(rbf, in.min.max.limits=c.minmax, abs.min.max=out.range, input.names=names(Boston)[1:n.in], output.names=names(Boston)[n.in+1])
CI.CU <- ciu.inka$explain(instance.values, ind.inputs.to.explain=c(1), montecarlo.samples = 100)
ciu.inka$barplot.CI.CU(instance.values)
#ciu$barplot.CI.CU(instance.values, neutral.CU=-0.001)

# CIU with Gradient Boosting
in.mins <- apply(Boston[,1:n.in], 2, min)
in.maxs <- apply(Boston[,1:n.in], 2, max)
c.minmax <- cbind(in.mins, in.maxs)
predict.function <- function(model, inputs) { predict(model,inputs,n.trees=10000) }
ciu <- ciu.new(Boston.boost, in.min.max.limits=c.minmax, abs.min.max=out.range, 
               input.names=names(Boston)[1:n.in], output.names=names(Boston)[n.in+1], 
               predict.function=predict.function)
inst.ind <- 1
CI.CU <- ciu$explain(Boston[inst.ind,], ind.inputs.to.explain=c(1), montecarlo.samples = 100)
ciu$barplot.CI.CU(Boston[inst.ind,1:n.in])

# CIU with Gradient Boosting, FunctionApproximator wrapper
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
inst.ind <- 1
CI.CU <- ciu.gbm.fa$explain(Boston[inst.ind,], ind.inputs.to.explain=c(1), montecarlo.samples = 100)
ciu.gbm.fa$barplot.CI.CU(Boston[inst.ind,1:n.in])

# Instance 406 has medv=5
# Instance 370 has medv=50
# Instance 6 has medv=28.7

# LIME: can't get to work yet...
# require(lime)
# predict_model.gbm <- function(x, newdata, type, ...) {
#   predict(x,newdata,n.trees=10000)
# }
# model_type.gbm <- function(x, ...) "regression"
# explainer <- lime(Boston[,1:13], Boston.boost)
# m<-Boston[c(406,6,370), 1:n.in]
# explanation <- explain(m, explainer, n_labels = 1, n_features = 13)
# print(explanation)
# plot_features(explanation)

