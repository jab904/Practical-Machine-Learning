# Practical Machine Learning Project
Judy  
March 30, 2016  
##Overview
For this assignment we will be building a prediction algorithm using data collected from the accelerometers on the belt, forearm, arm, and dumbbell of 6 participants.  The participants performed barbell lifts in five different ways, once correctly (classe A) and then in four other ways that represent common mistakes (classes B-E).  We will use their accelerometer data to create a prediction model that can be applied to another group of individuals (the test data), to determine the manner in which they performed the activities.  The final model used random forests, and predicted with 100% accuracy the 20 test samples. The data from this assignment was generously provide by http://groupware.les.inf.puc-rio.br/har.

##Getting started, loading the data
The training data can be downloaded at https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv, and the test data is available at https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv.


```r
setwd("C:/Users/JEB/Desktop/Coursera/Machine Learning")
suppressMessages(library(caret))
suppressMessages(library(rattle))
suppressMessages(library(randomForest))
data1<-read.csv("pml-training.csv", na.strings = c("NA", "", "#DIV/0!"))
data2<-read.csv("pml-testing.csv", na.strings = c("NA", "", "#DIV/0!"))
dim(data1)
```

```
## [1] 19622   160
```


##Cleaning and exploring the data
There are 160 variables.  We will need to narrow them down in order to generate the best model that is most predictive of the outcome. After exploring the data a little bit we can see that there are a lot of columns that can be cut. Let's start by looking at which columns have NAs in them since prediction algorithms often cannot handle missing data. (The results are hidden as they are extremely long).

```r
str(data1)
summary(data1)
colSums(is.na(data1))
```
Summing up all the NA's in each column shows us that some columns have no NA so we will keep those columns.  All the columns which contain NA values are missing over 97% of the data.  These will be eliminated since they don't provide us with any useful information. Next, we will eliminate any variable that have very little variance since they won't be good predictors. From the variables that are left we will select only the ones that contain information about the measurements of interest (belt, forearm, arm, and dumbbell).  Finally we will eliminate highly correlated variables as they will be redundant in our model.  

```r
col_of_int<-as.vector(which(colSums(is.na(data1))==0))
data1<-data1[,col_of_int]
zeroVar<-nearZeroVar(data1)
data1<-data1[,-zeroVar]
predictors1<-grep("arm|belt|bell|classe", names(data1))
data1<-data1[,predictors1]
df2=cor(data1[,-53])
highCor<-findCorrelation(df2,cutoff=0.9)
data1<-data1[,-highCor]
dim(data1)
```

```
## [1] 19622    46
```

```r
featurePlot(data1[,-46], y=data1$classe)
```

![](index_files/figure-html/unnamed-chunk-3-1.png)
We can see that the data cleaning worked very well since it significantly reduced the number of variables down to 46.  We can also see from the features plot that some variables look like they will be good predictors of the outcome while other will not be as useful.

##Splitting the data
Now we will split the training data into a training and testing set for cross validation purposes.

```r
set.seed(1)
inTrain<-createDataPartition(y=data1$classe, p=0.7, list=FALSE)
training<-data1[inTrain,]
testing<-data1[-inTrain,]
dim(testing)
```

```
## [1] 5885   46
```

```r
dim(training)
```

```
## [1] 13737    46
```

##Prediction with trees
The first model we will examine will use trees to predict the outcome.

```r
set.seed(2)
mod_rpart <- train(classe ~ .,method="rpart",data=training)
```

```
## Loading required package: rpart
```

```r
fancyRpartPlot(mod_rpart$finalModel)
```

![](index_files/figure-html/unnamed-chunk-5-1.png)

We see that this model splits up the data using the measurements from the pitch-forearm, magnet-belt-y, magnet-dumbell-y, roll-forearm, and accel-forearm-x variables.

###Cross validation for trees
Now we want to see how accurate the model that we built is.  We will apply it to our testing set and see how well it predicts the outcome. 

```r
confusionMatrix(testing$classe, predict(mod_rpart, testing))$overall["Accuracy"]
```

```
##  Accuracy 
## 0.4944775
```
We see that the accuracy of this method is only about 50% which is not very good at all.  We will now try another prediction method using random forests and see how it performs.

##Random Forests

```r
set.seed(3)
mod_rf<-randomForest(classe~.,data=training)
mod_rf
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 6
## 
##         OOB estimate of  error rate: 0.55%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3901    4    0    0    1 0.001280082
## B   12 2640    6    0    0 0.006772009
## C    0   15 2378    3    0 0.007512521
## D    0    0   26 2224    2 0.012433393
## E    0    0    2    5 2518 0.002772277
```

```r
order(varImp(mod_rf), decreasing=T)
```

```
##  [1]  2 32  1 34 31 33  8  9 30 22 28  6 10  3 29 40 45  7 24 25 42 19 26
## [24] 12 27 43 16 20 44 23 21 17 35 11 14 41 18 38  5  4 36 13 39 37 15
```
By ordering the variables in the random forest model, we see that the five most important variables are yaw-belt, magnet-dumbbell-z, pitch-forearm, pitch-belt, magnet-dumbbell-y. Some of these are the same variables selected from the tree model but there are some differences as well. We will do another random forest analysis using just these top 5 variables and see how the two models compare.

##Another random forest model and cross validation

```r
set.seed(4)
mod_rf2<-randomForest(classe~yaw_belt+magnet_dumbbell_z +pitch_forearm +pitch_belt+magnet_dumbbell_y,data=training)
confusionMatrix(testing$classe, predict(mod_rf, testing))$overall["Accuracy"]
```

```
##  Accuracy 
## 0.9959218
```

```r
confusionMatrix(testing$classe, predict(mod_rf2, testing))$overall["Accuracy"]
```

```
##  Accuracy 
## 0.9700935
```
The accuracy for the first random forest model (using all the variables) was 99.6%, making the out of sample error rate around 0.4%.  The accuracy for the second random forest model is 97% making its out of sample error rate around 3%. We see that both models are highly accurate.  When we include only the top 5 variables we lose a little accuracy but it is much faster computationally. I was planning on doing model stacking using trees and random forests but it is clear that won't be necessary. Since random forests alone give us extremely high accuracy we will not gain anything from stacking other models with it.

##Making predictions
Now we can use our final model to make predictions from the testing set.

```r
predict(mod_rf, data2)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

```r
predict(mod_rf2, data2)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
Not surprisingly, we see that both models give the same predictions. After submitting these predictions online we see it was 100% correct for the testing set.
