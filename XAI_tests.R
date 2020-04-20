# Code snippets for testing/running different XAI methods. 
#
# Created by Kary Främling 10apr2020
#

# Understanding lime, Thomas Lin Pedersen & Michaël Benesty: 
# https://cran.r-project.org/web/packages/lime/vignettes/Understanding_lime.html

# Lime package documentation: https://cran.r-project.org/web/packages/lime/lime.pdf

# Explaining a model and an explainer for it
library(MASS)
library(lime)
iris_test <- iris[1, 1:4]
iris_train <- iris[-1, 1:4]
iris_lab <- iris[[5]][-1]
model <- lda(iris_train, iris_lab)
explainer <- lime(iris_train, model)
# This can now be used together with the explain method
#explanation <- explain(iris_test, explainer, n_labels = 3, n_features = 4)
# The hypothetical Iris that we have been using in CIU tests.
iris_test[1,]<-c(7, 3.2, 6, 1.8)
explanation <- explain(iris_test, explainer, n_labels = 3, n_features = 4)
plot_features(explanation)

# Do same with CIU
source("ContextualImportanceUtility.R")
#predict(model,iris_test)
in.mins <- apply(iris_train, 2, min)
in.maxs <- apply(iris_train, 2, max)
c.minmax <- cbind(in.mins, in.maxs)
#bad.predict.function <- function(model, inputs) { c(1) } 
#good.lda.predict.function <- function(model, inputs) { pred <- predict(model,inputs); return(pred$posterior) }
#CI.CU <- contextual.IU(model, iris_test, c(1),matrix(c(0,1,0,1,0,1), ncol = 2, byrow = T), montecarlo.samples = 1000, c.minmax = c.minmax)
out.names <- levels(iris_lab)
ciu <- ciu.new(model, in.min.max.limits=c.minmax, abs.min.max=matrix(c(0,1,0,1,0,1), ncol = 2, byrow = T), 
               output.names=out.names)
#ciu <- ciu.new(model, in.min.max.limits=c.minmax, abs.min.max=matrix(c(0,1,0,1,0,1), ncol = 2, byrow = T), 
#               output.names=out.names, predict.function=good.lda.predict.function)
CI.CU <- ciu$explain(iris_test, ind.inputs.to.explain=c(1))
CI.CU
inp.ind.to.plot <- 3
out.ind.to.plot <- 2
ciu$plot.CI.CU(iris_test, ind.input=c(inp.ind.to.plot), ind.output=c(out.ind.to.plot), 
               n.points=40, xlab=colnames(iris)[inp.ind.to.plot], ylab=out.names[out.ind.to.plot])
ciu$plot.CI.CU.3D(iris_test, ind.inputs=c(3,4), ind.output=3, n.points=40)
ciu$barplot.CI.CU(inputs=iris_test, ind.inputs=NULL, ind.output=3, neutral.CU=0.5)

# Create Random Forest model on iris data
library(caret)
model <- train(iris_train, iris_lab, method = 'rf')
ciu <- ciu.new(model, in.min.max.limits=c.minmax, abs.min.max=matrix(c(0,1,0,1,0,1), ncol = 2, byrow = T), output.names=out.names)
CI.CU <- ciu$explain(iris_test, ind.inputs.to.explain=c(1))
CI.CU
inp.ind.to.plot <- 3
out.ind.to.plot <- 2
ciu$plot.CI.CU(iris_test, ind.input=c(inp.ind.to.plot), ind.output=c(out.ind.to.plot), n.points=40, xlab=colnames(iris)[inp.ind.to.plot], ylab=out.names[out.ind.to.plot])
ciu$plot.CI.CU.3D(iris_test, ind.inputs=c(3,4), ind.output=3, n.points=40)
ciu$barplot.CI.CU(inputs=iris_test, ind.inputs=NULL, ind.output=3, neutral.CU=0.5)
# Lime
explainer <- lime(iris_train, model)
explanation <- explain(iris_test, explainer, n_labels = 3, n_features = 4)
plot_features(explanation)

# Molner's iml package https://cran.r-project.org/web/packages/iml/vignettes/intro.html
