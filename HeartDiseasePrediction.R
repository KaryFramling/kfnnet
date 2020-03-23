# Code for the NN models from caret comes from https://rpubs.com/mbbrigitte/heartdisease

# Load necessary libraries
library(caret)
#library(mlbench)
library(pROC)

source("rbf.R")

# Strange, this function wasn't found anymore when re-running heart disease commands. But 
# not needed if started with clean session, strange!
convert.magic <- function(obj, types) {
  for (i in 1:length(obj)) {
    FUN <- switch(types[i], character = as.character, numeric = as.numeric, factor = as.factor)
    obj[, i] <- FUN(obj[, i])
  }
  obj
}

# Experiments with Heart Disease prediction data
heart.data <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",header=FALSE,sep=",",na.strings = '?')
names(heart.data) <- c( "age", "sex", "cp", "trestbps", "chol","fbs", "restecg",
                        "thalach","exang", "oldpeak","slope", "ca", "thal", "num")

# Only two classes, heart disease or not. Would it be worth having more? 
# Original data has 0-4 but no clue what is difference for 1-4.
heart.data$num[heart.data$num > 0] <- 1
# change a few predictor variables from integer to factors (make dummies)
chclass <-c("numeric","factor","factor","numeric","numeric","factor","factor","numeric","factor","numeric","factor","factor","factor","factor")
heart.data <- convert.magic(heart.data,chclass)

# Omit 6 rows with NA values.
heart.data <- na.omit(heart.data)

# Split into training and test set. Use fixed seed so that it always gives the same partition. 
set.seed(10)
inTrainRows <- createDataPartition(heart.data$num,p=0.7,list=FALSE)
trainData <- heart.data[inTrainRows,]
testData <-  heart.data[-inTrainRows,]

# Logistic regression worked the best for this case in tutorial. 
# There's something being messed up here between different libraries. This 
# should work (has worked) when starting off with empty work space and clean session. 
set.seed(10)
logRegModel <- train(num ~ ., data=trainData, method = 'glm', family = 'binomial')
logRegPrediction <- predict(logRegModel, testData)
logRegPredictionprob <- predict(logRegModel, testData, type='prob')[2] # Get the raw outputs
logRegConfMat <- confusionMatrix(logRegPrediction, testData[,"num"])
AUC = list()
Accuracy = list()
AUC$logReg <- roc(as.numeric(testData$num),as.numeric(as.matrix((logRegPredictionprob))))$auc
Accuracy$logReg <- logRegConfMat$overall['Accuracy']  #found names with str(logRegConfMat)

# Random Forest
library(randomForest)
set.seed(10)
RFModel <- randomForest(num ~ .,
                        data=trainData,
                        importance=TRUE,
                        ntree=2000)
#varImpPlot(RFModel)
RFPrediction <- predict(RFModel, testData)
RFPredictionprob = predict(RFModel,testData,type="prob")[, 2]
RFConfMat <- confusionMatrix(RFPrediction, testData[,"num"])
AUC$RF <- roc(as.numeric(testData$num),as.numeric(as.matrix((RFPredictionprob))))$auc
Accuracy$RF <- RFConfMat$overall['Accuracy'] 

# Boosted tree model with tuning (grid search)
# Boosted tree model (gbm) with adjusting learning rate and and trees.
set.seed(10)
objControl <- trainControl(method='cv', number=10,  repeats = 10)
gbmGrid <-  expand.grid(interaction.depth =  c(1, 5, 9),
                        n.trees = (1:30)*50,
                        shrinkage = 0.1,
                        n.minobsinnode =10)
# run model
boostModel <- train(num ~ .,data=trainData, method='gbm',
                    trControl=objControl, tuneGrid = gbmGrid, verbose=F)
# See model output in Appendix to get an idea how it selects best model
#trellis.par.set(caretTheme())
#plot(boostModel)
boostPrediction <- predict(boostModel, testData)
boostPredictionprob <- predict(boostModel, testData, type='prob')[2]
boostConfMat <- confusionMatrix(boostPrediction, testData[,"num"])
AUC$boost <- roc(as.numeric(testData$num),as.numeric(as.matrix((boostPredictionprob))))$auc
Accuracy$boost <- boostConfMat$overall['Accuracy']

#Stochastic gradient boosting
#This method finds tuning parameters automatically. But a bit more work to prepare data.
# for this to work add names to all levels (numbers not allowed)
feature.names=names(heart.data)
for (f in feature.names) {
  if (class(heart.data[[f]])=="factor") {
    levels <- unique(c(heart.data[[f]]))
    heart.data[[f]] <- factor(heart.data[[f]],
                              labels=make.names(levels))
  }
}
set.seed(10)
inTrainRows <- createDataPartition(heart.data$num,p=0.7,list=FALSE)
trainData2 <- heart.data[inTrainRows,]
testData2 <-  heart.data[-inTrainRows,]
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using
                           ## the following function
                           summaryFunction = twoClassSummary)

set.seed(10)
gbmModel <- train(num ~ ., data = trainData2,
                  method = "gbm",
                  trControl = fitControl,
                  verbose = FALSE,
                  tuneGrid = gbmGrid,
                  ## Specify which metric to optimize
                  metric = "ROC")
gbmPrediction <- predict(gbmModel, testData2)
gbmPredictionprob <- predict(gbmModel, testData2, type='prob')[2]
gbmConfMat <- confusionMatrix(gbmPrediction, testData2[,"num"])
#ROC Curve
AUC$gbm <- roc(as.numeric(testData2$num),as.numeric(as.matrix((gbmPredictionprob))))$auc
Accuracy$gbm <- gbmConfMat$overall['Accuracy']

# Support Vector Machine
set.seed(10)
svmModel <- train(num ~ ., data = trainData2,
                  method = "svmRadial",
                  trControl = fitControl,
                  preProcess = c("center", "scale"),
                  tuneLength = 8,
                  metric = "ROC")
svmPrediction <- predict(svmModel, testData2)
svmPredictionprob <- predict(svmModel, testData2, type='prob')[2]
svmConfMat <- confusionMatrix(svmPrediction, testData2[,"num"])
#ROC Curve
AUC$svm <- roc(as.numeric(testData2$num),as.numeric(as.matrix((svmPredictionprob))))$auc
Accuracy$svm <- svmConfMat$overall['Accuracy']  


# Train INKA with Heart Disease Prediction data.
heart.disease.inka <- function() {

  # Do necessary pre-processing here. First numeric values to [0,1]
  
  # Creating dummy variables is converting a categorical variable to as 
  # many binary variables as here are categories.
  dummies_model <- dummyVars(num ~ .,data=tD)
  tD <- predict(dummies_model, newdata = tD)

  # Do same for test set.
  testD <- predict(preProcess_range_model, newdata = testData)
  testD <- predict(dummies_model, newdata = testD)
  
  # Create in/out matrices
  t.in <- tD
  targets <- as.matrix(as.numeric(trainData$num)-1)
  n.in <- ncol(t.in)
  n.out <- 1
  #set.seed(10)
  rbf <-
    rbf.new(n.in,
            n.out,
            0,
            activation.function = squared.distance.activation,
            output.function = imqe.output.function)
  rbf$set.nrbf(TRUE)
  #aff.trans <- scale.translate.ranges(in.mins, in.maxs, c(0,0,0,0), c(1,1,1,1))
  ol <- rbf$get.outlayer()
  ol$set.use.bias(FALSE)
  rbf$set.spread(1) # d^2 parameter in INKA
  c <- 0.01 # The "c" parameter in INKA training, minimal distance for adding new hidden neuron.
  n.hidden <-
    train.inka(
      rbf,
      t.in,
      targets,
      c,
      max.iter = 50,
      inv.whole.set.at.end = F,
      classification.error.limit = 13
    )
  # Calculate error measure etc.
  y <- rbf$eval(t.in)
  #classification <- (y == apply(y, 1, max)) * 1
  nbr.errors <- sum(abs(targets - round(y))) # How many are mis-classified
  # confusionMatrix(as.factor(round(y)),as.factor(targets))
  # roc(targets,y)
  
  # Statistics for test data set.
  y.test <- rbf$eval(testD)
  targets.test <- as.matrix(as.numeric(testData$num)-1)
  nbr.errors.test <- sum(abs(targets.test - round(y.test)))
  inkaConfMat <<- confusionMatrix(as.factor(round(y.test)),as.factor(targets.test))
  Accuracy$inka <<- inkaConfMat$overall['Accuracy']  
  AUC$inka <<- roc(as.numeric(targets.test),as.numeric(y.test))$auc
  
  #cat("nbr.errors = ", nbr.errors, "\n")
  #cat("Number of hidden neurons: ", n.hidden, "\n")
  return(rbf) # Return trained RBF network
}

rbf.inka <- heart.disease.inka()

row.names <- names(Accuracy)
col.names <- c("AUC", "Accuracy")
cbind(as.data.frame(matrix(c(AUC,Accuracy),nrow = length(row.names), ncol = 2,
                           dimnames = list(row.names, col.names))))
