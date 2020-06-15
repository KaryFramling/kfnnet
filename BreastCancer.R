# Tests and utilities for Breast Cancer data set. 
#
# 699 observations on 11 variables, one being a character variable, 9 being ordered or nominal, 
# and 1 target class. 
# For caret models, see https://rpubs.com/ChristianLopezB/Supervised_Machine_Learning
#
# Created by Kary Främling 4oct2019
#

require(caret)

# Get Breast Cancer data set
require(mlbench)
data(BreastCancer)

# Remove rows with NAs. Remove ID attribute. 
BC <- BreastCancer[complete.cases(BreastCancer),-1]

# Create training and test sets
inTrain<-createDataPartition(y=BC$Class, p=0.75, list=FALSE) # 75% to train set
training.BreastCancer <- BC[inTrain,]
testing.BreastCancer <- BC[-inTrain,]
#preObj <- preProcess(training.BreastCancer[,-11], method = c("center", "scale"))
#preObjData <- predict(preObj,training.BreastCancer[,-11])
#modelFit<-train(Class~., data=training.BreastCancer, method="lda")
#Predict new data with model fitted
#predictions<-predict(modelFit, newdata=testing.BreastCancer)

#Shows Confusion Matrix and performance metrics
#confusionMatrix(predictions, testing.BreastCancer$Class)

kfoldcv <- trainControl(method="cv", number=10)
tc.none <- trainControl(method="none")
performance_metric <- "Accuracy"

#Linear Discriminant Analysis (LDA)
lda.BreastCancer <- train(Class~., data=training.BreastCancer, method="lda", metric=performance_metric, trControl=kfoldcv)
predictions <- predict(lda.BreastCancer, newdata=testing.BreastCancer)
confusionMatrix(predictions, testing.BreastCancer$Class)

#Classification and Regression Trees (CART)
cart.BreastCancer <- train(Class~., data=training.BreastCancer, method="rpart", metric=performance_metric, trControl=kfoldcv)
predictions <- predict(cart.BreastCancer, newdata=testing.BreastCancer)
confusionMatrix(predictions, testing.BreastCancer$Class)

#Support Vector Machines (SVM)
svm.BreastCancer <- train(Class~., data=training.BreastCancer, method="svmRadial", metric=performance_metric, trControl=kfoldcv,preProcess=c("center", "scale"))
predictions <- predict(svm.BreastCancer, newdata=testing.BreastCancer)
confusionMatrix(predictions, testing.BreastCancer$Class)

# Random Forest
rf.BreastCancer <- train(Class~., data=training.BreastCancer, method="rf", metric=performance_metric, trControl=kfoldcv,preProcess=c("center", "scale"))
predictions <- predict(rf.BreastCancer, newdata=testing.BreastCancer)
confusionMatrix(predictions, testing.BreastCancer$Class)

# # Summary of results
# results.BreastCancer <- resamples(list(lda=lda.BreastCancer, cart=cart.BreastCancer,  svm=svm.BreastCancer, rf=rf.BreastCancer))
# summary(results.BreastCancer)
# 
# # Plot results
# dotplot(results.BreastCancer)
