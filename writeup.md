# Prediction of Activities
Christopher Velayo  
Friday, October 24, 2014  

#Summary

The purpose of this assignment is to create a model based on a training set to predict how people performed barbell lifts based on accelerometer data. This model is then to be applied to a test set to determine the accuracy of the model. The information for this project came from the Human Activity Recognition dataset of the Groupware@LES group.  

#Preprocessing of Data

The course made available two CSV files with the needed data, pml-training.csv and pml-training.csv. The first step is to download the files and to read them into R.


```r
temptrainingdata <- read.csv("training.csv", stringsAsFactors = FALSE)
temptestingdata <- read.csv("testing.csv", stringsAsFactors = FALSE)
```

The next step is to perform some exploratory analyses to determine the suitability of the data.


```r
dim(temptrainingdata)
```

```
## [1] 19622   160
```

```r
dim(temptestingdata)
```

```
## [1]  20 160
```

Due to the number of blank and NA values, further exploration was done which showed that most of the blank and NA values came in a set of variables that only periodically had data. After determining the variables that consistently had values entered, subsets of the data were created containing only those variables. Also, the column containg the activity type was split off into its own vector.


```r
#Using grep to identify the variables to keep and creating a column index. 
numeric <- which(as.logical(grepl("^total_accel.*", names(temptrainingdata))+
                                grepl("^accel.*", names(temptrainingdata))+
                                grepl("^gyros.*", names(temptrainingdata))+
                                grepl("^magnet.*", names(temptrainingdata))+
                                grepl("^roll.*", names(temptrainingdata))+
                                grepl("^pitch.*", names(temptrainingdata))+
                                grepl("^yaw.*", names(temptrainingdata))))
#Using the column index to subset the testing and traing data
subset <- temptrainingdata[,numeric]
testing <- temptestingdata[,numeric]

#Creating a vector with the true answers
subset$classe <- as.factor(temptrainingdata$classe)
```

#Partitioning

To assess the accuracy of the model prior to applying the model to the testing data, the decision was made to create a cross validation data set. This was done using the createDataPartition function from the caret package. To minimize the chances of overfitting, the training data was split with 60% assigned to develop the model, and 40% assigned to cross validate the model.


```r
# Setting seed to ensure reproducibility
set.seed(3743)
# Loading caret package
suppressMessages(library(caret))

#Creating an index to split the data 60/40, then using the index to create the testing and
#cross validation sets
cvIndex = createDataPartition(subset$classe, p = 0.60,list=FALSE)
training <- subset[cvIndex,]
crossval <- subset[-cvIndex,]
```

#Creating models to choose from

To help increase the chances of an accurate model, several model types were used by caret to train the prediction model. I chose to use the Random Forest model, the Decision Tree model, and the Support Vector Machine with Linear Kernel model. Below are the results of each model.


```r
#load required libraries
suppressMessages(library(rpart))
suppressMessages(library(randomForest))
suppressMessages(library(kernlab))

#Random Forest Model
set.seed(23456)
rfmodel <- train(classe~., method="rf", data=training)
rfmodel$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 0.75%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3343    3    0    0    2    0.001493
## B   17 2256    6    0    0    0.010092
## C    0   18 2034    2    0    0.009737
## D    0    0   32 1896    2    0.017617
## E    0    0    1    5 2159    0.002771
```

```r
#Decision Tree Model
set.seed(5769837)
treemodel <- train(classe~., method="rpart", data=training)
treemodel
```

```
## CART 
## 
## 11776 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 11776, 11776, 11776, 11776, 11776, 11776, ... 
## 
## Resampling results across tuning parameters:
## 
##   cp    Accuracy  Kappa  Accuracy SD  Kappa SD
##   0.04  0.5       0.36   0.02         0.03    
##   0.05  0.4       0.21   0.06         0.10    
##   0.11  0.3       0.06   0.04         0.06    
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.03916.
```

```r
#Support Vector Machine Model
set.seed(38792)
suppressWarnings(svmmodel <- train(classe~., method="svmLinear", data=training))
svmmodel$finalModel
```

```
## Support Vector Machine object of class "ksvm" 
## 
## SV type: C-svc  (classification) 
##  parameter : cost C = 1 
## 
## Linear (vanilla) kernel function. 
## 
## Number of Support Vectors : 6286 
## 
## Objective Function Value : -1239 -1120 -948.3 -564.3 -1171 -749.8 -1510 -1058 -901 -1020 
## Training error : 0.211447
```

#Initial results

Based on the accuracy of the various models, the random forest model presented with the highest accuracy and an estimated out of sample error rate of 0%. In comparison, the support vector machine model has an estimated out of sample error rate of 21.1%, and the decision tree model has an estimated out of sample error rate of 52%. Especially with the very high accuracy for the random forest model, these models were applied to the cross validation data to check the estimates.


```r
#Use models to predict activity quality based on cross-validation dataset
rfcrossval <- predict(rfmodel, crossval)
treecrossval <- predict(treemodel, crossval)
svmcrossval <- predict(svmmodel, crossval)

#Compute accuracy
rfcvacc <- sum(crossval$classe == rfcrossval)/length(crossval$classe)
rfcvacc
```

```
## [1] 0.9915
```

```r
treecvacc <- sum(crossval$classe == treecrossval)/length(crossval$classe)
treecvacc
```

```
## [1] 0.4828
```

```r
svmcvacc <- sum(crossval$classe == svmcrossval)/length(crossval$classe)
svmcvacc
```

```
## [1] 0.7831
```

```r
#Compute difference from training set prediction and crossvalidation prediction
#Values multiplied by 100 to convert from decimal to percent
(rfcvacc - sum(training$classe == predict(rfmodel,training))/length(training$classe))*100
```

```
## [1] -0.8539
```

```r
(treecvacc - sum(training$classe == predict(treemodel,training))/length(training$classe))*100
```

```
## [1] 0.275
```

```r
(svmcvacc - sum(training$classe == predict(svmmodel,training))/length(training$classe))*100
```

```
## [1] -0.5479
```

Comparing these results to the estimates, we can see that all three models performed within 1% of their training set accuracy. While performing a principal component analysis was considered, with the already very high accuracy, the decision was made to not do the analysis as the potential gain seemed minimal.


```r
confusionMatrix(rfcrossval, crossval$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2229    6    0    0    0
##          B    3 1507   11    0    0
##          C    0    5 1355   36    0
##          D    0    0    2 1250    4
##          E    0    0    0    0 1438
## 
## Overall Statistics
##                                         
##                Accuracy : 0.991         
##                  95% CI : (0.989, 0.993)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.989         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.999    0.993    0.990    0.972    0.997
## Specificity             0.999    0.998    0.994    0.999    1.000
## Pos Pred Value          0.997    0.991    0.971    0.995    1.000
## Neg Pred Value          0.999    0.998    0.998    0.995    0.999
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.192    0.173    0.159    0.183
## Detection Prevalence    0.285    0.194    0.178    0.160    0.183
## Balanced Accuracy       0.999    0.995    0.992    0.986    0.999
```

#Analysis of model

Based on the crossvalidation confusion matrix, the model appears to have the hardest time classifying C activities and the easiest time classifying E activities. The most common mistakes were classifying C activities as D activities, and classifying B activities as C activities. Further exploration around these mistakes may identify interesting issues in the data. It is also interesting that the confusion matrix based on the training data differed from the confusion matrix with the cross-validation data.

#Results

After using the model to predict the activities based on the test data, these predictions were submitted to the Coursera website. All predicitons were correct based on the autograder. In comparison, the decision tree model got 6 predictions correctly, and the support vector machine model got 15 predictions correct, making it clear that the random forest model would the correct one to choose.
