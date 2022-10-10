# ------------------------------------------------------------------------------
## '-----------------'  AD vs CTL Classification Problem  '-----------------' ##
# ------------------------------------------------------------------------------


## -------  Librairies  -------
library(caret)  # Used for Classification & Regression Training
library(reshape2)
library(ggplot2)  # Used for plotting
library(hrbrthemes)  # Used for plot themes
library(mlbench)  # Used for some metrics
library(MLmetrics)  # Used for some metrics
library(mltools)  # Used for computing MCC 
library(pROC)  # Used for computing AUC
library(outliers)  # Used for Grubbs Test
library(EnvStats)  # Used for Rosner Test


## -------  Working Directory  -------
getwd()  # Get the current working directory
setwd("C:/Users/PC/OneDrive/Bureau/MaIA/Statistical Learning & Data Mining/Assignment2")  # Set to current Working Directory if needed

train_data_ <- read.csv('ADCTLtrain.csv')  # Read Train Data
test_data_ <- read.csv('ADCTLtest.csv')  # Read Test Data

## -------  Remove the index column 'ID'  -------
train_data = subset(train_data_, select = -ID)
test_data = subset(test_data_, select = -ID)
#head(train_data)  # Show some Data values
dim(train_data)  # Check that the dimension is correct (164x430)

attach(train_data)
options(digits = 3)  # Set number of digits in float
prop.table(table(train_data$Label))  # We can see that the classes are well balanced

x_train_data = subset(train_data, select = -Label)  # Removing Labels from the vector of features


# ------------------------------------------------------------------------------
# -------------------------  FINDING OUTLIERS  ---------------------------------

# ---  Outliers Through Grubbs Test  --------------------------------
# We define the following function to find potential outliers (whose p-value<0.05) through the 'grubbs.test' built-in function
getGrubbsOutliers <- function(xTrain) {
  grubbsInd <- NULL  # Empty vector to store outliers' indexes
  for(i in 1:NCOL(xTrain)) {
    test1 <- (grubbs.test(xTrain[, i], type=10))$p.value  # Get the max value's p.value
    if (test1 < 0.05)
      grubbsInd <- append(grubbsInd, which(xTrain[, i] == max(xTrain[, i])))
    test2 <- (grubbs.test(xTrain[, i], type=10, opposite = TRUE))$p.value  # Get the min value's p.value
    if (test2 < 0.05)
      grubbsInd <- append(grubbsInd, which(xTrain[, i] == min(xTrain[, i])))
  }
  return (grubbsInd)
}
outlGrubbs <- getGrubbsOutliers(x_train_data)
indGrubbs = outlGrubbs[!duplicated(outlGrubbs)]  # Remove duplicated outliers
length(indGrubbs)  # Check the final number of outliers found
outGrubbs_train = train_data[-indGrubbs,]  # Remove those outliers from the Train Set
dim(outGrubbs_train)  # Check the new dimension of the Train Set

# ---  Outliers Through Rosner Test  --------------------------------
# We define the following function to find potential outliers through the 'rosnerTest' built-in function
getRosnerOutliers <- function(xTrain) {
  rosnerInd <- NULL  # Empty vector to store outliers' indexes
  for(i in 1:NCOL(xTrain)) {
    test <- rosnerTest(xTrain[, i], k = 3)
    val <- (test$all.stats$Value[(which(test$all.stats$Outlier == 'TRUE'))])  # Check the values that are considered TRUE outliers
    rosnerInd <- append(rosnerInd, which(xTrain[, i] == val ))  # Get the indexes of those values
  }
  return (rosnerInd)
}
outlRosner <- getRosnerOutliers(x_train_data)
indRosner = outlRosner[!duplicated(outlRosner)]  # Remove duplicated outliers
length(indRosner)  # Check the final number of outliers found
outRosner_train = train_data[-indRosner,]  # Remove those outliers from the Train Set
dim(outRosner_train)  # Check the new dimension of the Train Set

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
x_train_data = subset(train_data, select = -Label);  # Initial Feature vector
x_train_data_grubbs = subset(outGrubbs_train, select = -Label);  # Feature vector without outliers (Grubbs)
x_train_data_rosner = subset(outRosner_train, select = -Label);  # Feature vector without outliers (Rosner)

# ------------------------------------------------------------------------------
# -----------------  REMOVING HIGHLY CORRELATED FEATURES  ----------------------
# Remove redundant features (Highly Correlated features) from the whole Train Set
set.seed(7)
correlationMatrix <- cor(x_train_data)  # calculate correlation matrix
print(correlationMatrix)  # summarize the correlation matrix
cormat <- round(cor(x_train_data),2)
melted_cormat <- melt(cormat)  # Flatten the correlation matrix
# Plot the Heatmap for the Correlation Matrix
trellis.device()
ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + ggtitle('Correlation Heatmap for ADvsCTL') +
  geom_tile() + theme(axis.text.x=element_blank(),axis.text.y=element_blank()) +
  scale_fill_distiller(palette = "RdPu", direction = 1) 
#theme_ipsum()

# Find and remove attributes that are highly corrected (cutoff >= 0.9) :
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.9)
length(highlyCorrelated)  # Check the number of highly correlated features
reduced_train = subset(train_data, select = -highlyCorrelated)  # Removing those features

x_reduced_train = subset(reduced_train, select = -Label)  # New Feature vector with reduced features
# Plot the new Heatmap of Correlation Matrix :
cormat2 <- round(cor(x_reduced_train),2)
melted_cormat2 <- melt(cormat2)
ggplot(data = melted_cormat2, aes(x=Var1, y=Var2, fill=value)) +
  geom_tile() +
  scale_fill_distiller(palette = "RdPu", direction = 1) +
  theme_ipsum()

## --->  We can apply the same steps for the Data without outliers, but the results were pretty similar..

# Remove redundant features (Highly Correlated features) from the Grubbs' Outliers-removed Train Set
set.seed(7)
correlationMatrix <- cor(x_train_data_grubbs)  # calculate correlation matrix
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.9)
reduced_train_grubbs = subset(outGrubbs_train, select = -highlyCorrelated)
dim(reduced_train_grubbs)
x_reduced_train_grubbs = subset(reduced_train_grubbs, select = -Label)

# Remove redundant features (Highly Correlated features) from the Rosner's Outliers-removed Train Set
set.seed(7)
correlationMatrix <- cor(x_train_data_rosner)  # calculate correlation matrix
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.9)
reduced_train_rosner = subset(outRosner_train, select = -highlyCorrelated)
dim(reduced_train_rosner)
x_reduced_train_rosner = subset(reduced_train_rosner, select = -Label)


# ------------------------------------------------------------------------------
# ---------------------  RECURSIVE FEATURE ELIMINATION  ------------------------

y_train = (factor(train_data$Label))  # Factor the labels
# Define the control function 'rfeControl' for 10 folds
set.seed(7)
ctrl <- rfeControl(method = "repeatedcv", repeats = 5, number=10,
                   saveDetails = TRUE, returnResamp = "final")

# Summary Functions to define the metrics to be used..
mccSummary <- function (data, lev = NULL, model = NULL) {
  out <- mcc(data$obs, data$pred)
  names(out) <- "MCC"
  out }
aucSummary <- function (data, lev = NULL, model = NULL) {
  out <- auc(as.numeric(data$obs), as.numeric(data$pred))
  names(out) <- "pAUC"
  out }

# Summary function for the RFE Control, combining different metrics
metricStats <- function(...) c(twoClassSummary(...), defaultSummary(...), prSummary(...), mccSummary(...), aucSummary(...))
ctrl$functions$summary <- metricStats


## ---> From here, we will try different classifiers with the same approach for
# each one of them, and according to their results, we will choose our final Model for prediction


## -----------------------------------------------------------------------------
## ----------------  LINEAR DISCRIMINANT ANALYSIS (LDA)  -----------------------
## -----------------------------------------------------------------------------

# ---  Define the RFE function for LDA  ---
ctrl$functions <- ldaFuncs
ctrl$functions$summary <- metricStats
set.seed(7)
# Run the RFE Algorithm, whose performance is based on AUC metric, for different sizes (number of possible features) :
ldaRFE <- rfe(x=x_train_data, y=y_train, metric = "pAUC",
              sizes = c(4,8,10,12,14,16,18,20,22,24,50,100,200,300,400),
              preProc = c("center", "scale"), rfeControl = ctrl)
ldaRFE  # Show the results
plot(ldaRFE, type=c("g", "o"))  # Plot the AUC measures for different number of variables
ldaPredict = predictors(ldaRFE)  # Get the retained predictors
x_lda = subset(train_data, select = ldaPredict)  # Keep only the RFE features for the Train Set

##  --->  We do the same steps for the Outliers-removed Data (Both for Grubbs and Rosner) :
set.seed(7)  # Grubbs - Removed outliers
ldaRFE_grubbs <- rfe(x=x_train_data_grubbs, y=factor(outGrubbs_train$Label),
                     sizes = c(4,8,10,12,14,16,18,20,22,24,50,100,200,300,400), 
                     metric = "pAUC", preProc = c("center", "scale"), rfeControl = ctrl)
ldaPredict_grubbs = predictors(ldaRFE_grubbs)
# plot(ldaRFE_grubbs, type=c("g", "o"))  # Plot the AUC measures for different number of variables
x_lda_grubbs = subset(outGrubbs_train, select = ldaPredict_grubbs)

set.seed(7)  # Rosner - Removed outliers
ldaRFE_rosner <- rfe(x=x_train_data_rosner, y=factor(outRosner_train$Label),
                     sizes = c(4,8,10,12,14,16,18,20,22,24,50,100,150,200,250,300,350,400,420), 
                     metric = "pAUC", preProc = c("center", "scale"), rfeControl = ctrl)
ldaPredict_rosner = predictors(ldaRFE_rosner)
# plot(ldaRFE_rosner, type=c("g", "o"))  # Plot the AUC measures for different number of variables
x_lda_rosner = subset(outRosner_train, select = ldaPredict_rosner)


# ------------------------------------------------------------------------------
##  ---------------   Comparing different LDA Trainings   ----------------------

# -----------  LDA on RFE Train Data :  -----------
# The following part will be applied on the whole Train Set :
set.seed(7)
lda_results_rfe <- NULL  # Vector to store all metrics results
# Define the trainControl function for Caret's 'train'
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_lda_fullRFE = train(y = y_train, x = x_lda, method="lda",
                          preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_lda_fullRFE$resample)-1))){  # Get the mean result
  lda_results_rfe <- cbind(lda_results_rfe, mean(model_lda_fullRFE$resample[, i]))
}
colnames(lda_results_rfe) <- names((model_lda_fullRFE$resample)[1:(length(model_lda_fullRFE$resample)-1)])
lda_results_rfe  # Show the result metrics


# The following part will be applied on the Grubbs' Outliers-removed Train Set :
set.seed(7)
lda_results_rfe_grubbs <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_lda_grubbsRFE = train(y = outGrubbs_train$Label, x = x_lda_grubbs, method="lda",
                            preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_lda_grubbsRFE$resample)-1))){  # Get the mean result
  lda_results_rfe_grubbs <- cbind(lda_results_rfe_grubbs, mean(model_lda_grubbsRFE$resample[, i]))
}
colnames(lda_results_rfe_grubbs) <- names((model_lda_grubbsRFE$resample)[1:(length(model_lda_grubbsRFE$resample)-1)])
lda_results_rfe_grubbs  # Show the result metrics


# The following part will be applied on the Rosner' Outliers-removed Train Set :
set.seed(7)
lda_results_rfe_rosner <- NULL
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_lda_rosnerRFE = train(y = outRosner_train$Label, x = x_lda_rosner, method="lda",
                            preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_lda_rosnerRFE$resample)-1))){  # Get the mean result
  lda_results_rfe_rosner <- cbind(lda_results_rfe_rosner, mean(model_lda_rosnerRFE$resample[, i]))
}
colnames(lda_results_rfe_rosner) <- names((model_lda_rosnerRFE$resample)[1:(length(model_lda_rosnerRFE$resample)-1)])
lda_results_rfe_rosner  # Show the result metrics


# -----------  LDA on Full Train Data :  -----------
set.seed(7)
lda_results_f <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_lda_full = train(y = y_train, x = x_train_data, method="lda",
                       preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_lda_full$resample)-1))){  # Get the mean result
  lda_results_f <- cbind(lda_results_f, mean(model_lda_full$resample[, i]))
}
colnames(lda_results_f) <- names((model_lda_full$resample)[1:(length(model_lda_full$resample)-1)])
lda_results_f  # Show the result metrics


# -----------  LDA on Outliers-removed Train Data :  -----------

# ------- Grubbs Outliers-removed Train set : -------
set.seed(7)
lda_results_grubbs <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_lda_grubbs = train(y = outGrubbs_train$Label, x = outGrubbs_train[, -length(train_data[,])], method="lda",
                         preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_lda_grubbs$resample)-1))){
  lda_results_grubbs <- cbind(lda_results_grubbs, mean(model_lda_grubbs$resample[, i]))
}
colnames(lda_results_grubbs) <- names((model_lda_grubbs$resample)[1:(length(model_lda_grubbs$resample)-1)])
lda_results_grubbs  # Show the result metrics


# ------- Rosner Outliers-removed Train set : -------
set.seed(7)
lda_results_rosner <- NULL  # Vector to store all metrics results
yt <- (outRosner_train['Label'])  # Label vector
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_lda_rosner = train(y = factor(yt[,]), x = outRosner_train[, -length(train_data[,])], method="lda",
                         preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_lda_rosner$resample)-1))){
  lda_results_rosner <- cbind(lda_results_rosner, mean(model_lda_rosner$resample[, i]))
}
colnames(lda_results_rosner) <- names((model_lda_rosner$resample)[1:(length(model_lda_rosner$resample)-1)])
lda_results_rosner  # Show the result metrics


# -----------  LDA on Correlation-removed Train Data :  -----------
# Full Train Data (with outliers)
set.seed(7)
lda_results_r <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_lda_noCorr = train(y = y_train, x = x_reduced_train, method="lda",
                         preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_lda_noCorr$resample)-1))){
  lda_results_r <- cbind(lda_results_r, mean(model_lda_noCorr$resample[, i]))
}
colnames(lda_results_r) <- names((model_lda_noCorr$resample)[1:(length(model_lda_noCorr$resample)-1)])
lda_results_r  # Show the result metrics


# Grubbs - Train Data (without outliers and no correlation)
set.seed(7)
lda_results_rGrubbs <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_lda_grubbsNoCorr = train(y = outGrubbs_train$Label, x = x_reduced_train_grubbs, method="lda",
                               preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_lda_grubbsNoCorr$resample)-1))){
  lda_results_rGrubbs <- cbind(lda_results_rGrubbs, mean(model_lda_grubbsNoCorr$resample[, i]))
}
colnames(lda_results_rGrubbs) <- names((model_lda_grubbsNoCorr$resample)[1:(length(model_lda_grubbsNoCorr$resample)-1)])
lda_results_rGrubbs  # Show the result metrics


# Rosner - Train Data (without outliers and no correlation)
set.seed(7)
lda_results_rRosner <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_lda_rosnerNoCorr = train(y = outRosner_train$Label, x = x_reduced_train_rosner, method="lda",
                               preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_lda_rosnerNoCorr$resample)-1))){
  lda_results_rRosner <- cbind(lda_results_rRosner, mean(model_lda_rosnerNoCorr$resample[, i]))
}
colnames(lda_results_rRosner) <- names((model_lda_rosnerNoCorr$resample)[1:(length(model_lda_rosnerNoCorr$resample)-1)])
lda_results_rRosner  # Show the result metrics


# -----------  LDA on PCA Train Data :  -----------

# The following part will apply PCA on the whole Train Set, with different thresholds 
set.seed(7)
lda_results_pca <- NULL  # Vector to store all metrics results
for(i in seq(70,95,5)){  # Looping over different thresholds for PCA
  control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                          preProcOptions = list(thresh = i/100), classProbs = TRUE, summaryFunction = metricStats)
  model_lda_fullPCA = train(y = y_train, x = x_train_data, method="lda",
                            preProcess=c("center", "scale", "pca"), trControl = control)
  lda_results_pca <- rbind(lda_results_pca, cbind(model_lda_fullPCA$results$pAUC, model_lda_fullPCA$results$MCC, model_lda_fullPCA$results$ROC, model_lda_fullPCA$results$Sens, model_lda_fullPCA$results$Spec, model_lda_fullPCA$results$Accuracy, model_lda_fullPCA$results$Kappa, model_lda_fullPCA$results$AUC, model_lda_fullPCA$results$Precision, model_lda_fullPCA$results$Recall, model_lda_fullPCA$results$F))
  colnames(lda_results_pca) <- cbind('pAUC', 'MCC', 'ROC', 'Sens', 'Spec', 'Acc', 'Kappa', 'AUC', 'Precision', 'Recall', 'F') }
lda_results_pca <- lda_results_pca[which.max(lda_results_pca[,1]),]  # Keep the one with max pAUC amoung the different thresholds
lda_results_pca # Show the result metrics


# The following part will apply PCA on the Correlation-removed Train Set, with different thresholds :
set.seed(7)
lda_results_rpca <- NULL  # Vector to store all metrics results
for(i in seq(70,95,5)){  # Looping over different thresholds for PCA
  control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                          preProcOptions = list(thresh = i/100), classProbs = TRUE, summaryFunction = metricStats)
  model_lda_rPCA = train(y = y_train, x = x_reduced_train, method="lda",
                         preProcess=c("center", "scale", "pca"), trControl = control)
  lda_results_rpca <- rbind(lda_results_rpca, cbind(model_lda_rPCA$results$pAUC, model_lda_rPCA$results$MCC, model_lda_rPCA$results$ROC, model_lda_rPCA$results$Sens, model_lda_rPCA$results$Spec, model_lda_rPCA$results$Accuracy, model_lda_rPCA$results$Kappa, model_lda_rPCA$results$AUC, model_lda_rPCA$results$Precision, model_lda_rPCA$results$Recall, model_lda_rPCA$results$F))
  colnames(lda_results_rpca) <- cbind('pAUC', 'MCC', 'ROC', 'Sens', 'Spec', 'Acc', 'Kappa', 'AUC', 'Precision', 'Recall', 'F') }
lda_results_rpca <- lda_results_rpca[which.max(lda_results_rpca[,1]),]  # Keep the one with max pAUC aòoung the different thresholds
lda_results_rpca  # Show the result metrics

# The following part will apply PCA on the Grubbs Outliers-removed Train Set, with different thresholds :
set.seed(7)
lda_results_rpca_grubbs <- NULL  # Vector to store all metrics results
for(i in seq(70,95,5)){  # Looping over different thresholds for PCA
  control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                          preProcOptions = list(thresh = i/100), classProbs = TRUE, summaryFunction = metricStats)
  model_lda_grubbsPCA = train(y = outGrubbs_train$Label, x = x_reduced_train_grubbs, method="lda",
                              preProcess=c("center", "scale", "pca"), trControl = control)
  lda_results_rpca_grubbs <- rbind(lda_results_rpca_grubbs, cbind(model_lda_grubbsPCA$results$pAUC, model_lda_grubbsPCA$results$MCC, model_lda_grubbsPCA$results$ROC, model_lda_grubbsPCA$results$Sens, model_lda_grubbsPCA$results$Spec, model_lda_grubbsPCA$results$Accuracy, model_lda_grubbsPCA$results$Kappa, model_lda_grubbsPCA$results$AUC, model_lda_grubbsPCA$results$Precision, model_lda_grubbsPCA$results$Recall, model_lda_grubbsPCA$results$F))
  colnames(lda_results_rpca_grubbs) <- cbind('pAUC', 'MCC', 'ROC', 'Sens', 'Spec', 'Acc', 'Kappa', 'AUC', 'Precision', 'Recall', 'F') }
lda_results_rpca_grubbs <- lda_results_rpca_grubbs[which.max(lda_results_rpca_grubbs[,1]),]  # Keep the one with max pAUC aòoung the different thresholds
lda_results_rpca_grubbs  # Show the result metrics

# The following part will apply PCA on the Rosner Outliers-removed Train Set, with different thresholds :
set.seed(7)
lda_results_rpca_rosner <- NULL  # Vector to store all metrics results
for(i in seq(70,95,5)){  # Looping over different thresholds for PCA
  control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                          preProcOptions = list(thresh = i/100), classProbs = TRUE, summaryFunction = metricStats)
  model_lda_rosnerPCA = train(y = outRosner_train$Label, x = x_reduced_train_rosner, method="lda",
                              preProcess=c("center", "scale", "pca"), trControl = control)
  lda_results_rpca_rosner <- rbind(lda_results_rpca_rosner, cbind(model_lda_rosnerPCA$results$pAUC, model_lda_rosnerPCA$results$MCC, model_lda_rosnerPCA$results$ROC, model_lda_rosnerPCA$results$Sens, model_lda_rosnerPCA$results$Spec, model_lda_rosnerPCA$results$Accuracy, model_lda_rosnerPCA$results$Kappa, model_lda_rosnerPCA$results$AUC, model_lda_rosnerPCA$results$Precision, model_lda_rosnerPCA$results$Recall, model_lda_rosnerPCA$results$F))
  colnames(lda_results_rpca_rosner) <- cbind('pAUC', 'MCC', 'ROC', 'Sens', 'Spec', 'Acc', 'Kappa', 'AUC', 'Precision', 'Recall', 'F') }
lda_results_rpca_rosner <- lda_results_rpca_rosner[which.max(lda_results_rpca_rosner[,1]),]  # Keep the one with max pAUC aòoung the different thresholds
lda_results_rpca_rosner  # Show the result metrics


# -------  Summary of the different results :  -------

res_lda <- rbind(lda_results_f, lda_results_grubbs, lda_results_rosner,
                 lda_results_r, lda_results_rGrubbs,lda_results_rRosner,
                 lda_results_rfe, lda_results_rfe_grubbs,lda_results_rfe_rosner)
rownames(res_lda) <- cbind('Full', 'Grubbs', 'Rosner', 'FullNoCorr', 'GrubbsNoCorr', 'rosnerNoCorr',
                           'FullRFE', 'GrubbsRFE', 'RosnerRFE')
res_lda <- res_lda[, c(11,10,1,2,3,4,5,6,7,8,9)]  # Reorder the columns to compare
resPca_lda <- rbind(lda_results_pca, lda_results_rpca, lda_results_rpca_grubbs, lda_results_rpca_rosner)
rownames(resPca_lda) <- cbind('FullPCA', 'ReducedPCA', 'GrubbsPCA', 'RosnerPCA')
# Show results :
scores_lda <- rbind(res_lda, resPca_lda)
scores_lda


# # ## ----  FINAL RESULTS - LDA :  ----
#              pAUC   MCC    ROC   Sens  Spec  Accuracy Kappa AUC   Precision Recall F
# Full         0.848  0.702  0.919 0.823 0.873 0.848    0.695 0.807 0.868     0.823  0.840
# Grubbs       0.811  0.629  0.906 0.818 0.804 0.809    0.617 0.701 0.787     0.818  0.792
# Rosner       0.849  0.708  0.938 0.828 0.871 0.850    0.699 0.804 0.858     0.828  0.837
# FullNoCorr   0.817  0.643  0.884 0.798 0.836 0.817    0.634 0.769 0.836     0.798  0.810
# GrubbsNoCorr 0.786  0.581  0.892 0.804 0.768 0.783    0.566 0.690 0.756     0.804  0.768
# rosnerNoCorr 0.824  0.663  0.915 0.803 0.846 0.826    0.649 0.776 0.836     0.803  0.808
# FullRFE      0.874  0.757  0.936 0.847 0.900 0.875    0.748 0.804 0.899     0.847  0.866
# GrubbsRFE    0.876  0.761  0.938 0.871 0.881 0.875    0.748 0.734 0.869     0.871  0.860
# RosnerRFE    0.870  0.750  0.944 0.865 0.875 0.870    0.739 0.806 0.874     0.865  0.862
# FullPCA      0.887  0.784  0.954 0.877 0.898 0.888    0.776 0.825 0.902     0.877  0.885
# ReducedPCA   0.886  0.779  0.930 0.913 0.859 0.886    0.772 0.798 0.873     0.913  0.889
# GrubbsPCA    0.832  0.691  0.941 0.828 0.837 0.833    0.665 0.753 0.848     0.828  0.817
# RosnerPCA    0.877  0.767  0.963 0.845 0.908 0.879    0.755 0.817 0.903     0.845  0.864
  

# Best LDA :
# FullPCA      0.887  0.784  0.954 0.877 0.898 0.888    0.776 0.825 0.902     0.877  0.885
bestModel_LDA <- model_lda_fullPCA

# ---------------------------------------------------------------------------------------------------


## -----------------------------------------------------------------------------
## ---------------------  K_NEAREST NEIGHBORS (KNN)  ---------------------------
## -----------------------------------------------------------------------------

# # ---  Define the RFE function for KNN  ---
ctrl$functions <- caretFuncs
ctrl$functions$summary <- metricStats
set.seed(7)
# # Run the RFE Algorithm, whose performance is based on AUC metric, for different sizes (number of possible features) :
knnRFE <- rfe(x=x_train_data, y=y_train, metric = "pAUC", method = 'knn', 
              sizes = 2^(2:8), tuneLength=10, 
              preProc = c("center", "scale"), rfeControl = ctrl)
knnRFE  # Show the results
plot(knnRFE, type=c("g", "o"))  # Plot the AUC measures for different number of variables
knnPredict = predictors(knnRFE)  # Get the retained predictors
x_knn = subset(train_data, select = knnPredict)  # Keep only the RFE features for the Train Set

# ##  --->  We do the same steps for the Outliers-removed Data (Both for Grubbs and Rosner) :
set.seed(7)  # Grubbs - Removed outliers
knnRFE_grubbs <- rfe(x=x_train_data_grubbs, y=factor(outGrubbs_train$Label), method = 'knn',
                     sizes = 2^(2:8), tuneLength=10, 
                     metric = "pAUC", preProc = c("center", "scale"), rfeControl = ctrl)
knnPredict_grubbs = predictors(knnRFE_grubbs)
# plot(knnRFE_grubbs, type=c("g", "o"))  # Plot the AUC measures for different number of variables
x_knn_grubbs = subset(outGrubbs_train, select = knnPredict_grubbs)

set.seed(7)  # Rosner - Removed outliers
knnRFE_rosner <- rfe(x=x_train_data_rosner, y=factor(outRosner_train$Label), method = 'knn',
                     sizes = 2^(2:8), tuneLength=10, 
                     metric = "pAUC", preProc = c("center", "scale"), rfeControl = ctrl)
knnPredict_rosner = predictors(knnRFE_rosner)
# plot(knnRFE_rosner, type=c("g", "o"))  # Plot the AUC measures for different number of variables
x_knn_rosner = subset(outRosner_train, select = knnPredict_rosner)


# # ------------------------------------------------------------------------------
# ##  ---------------   Comparing different KNN Trainings   ----------------------
#
# # -----------  KNN on RFE Train Data :  -----------
# # The following part will be applied on the whole Train Set :
set.seed(7)
knn_results_rfe <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats)
model_knn_fullRFE = train(y = y_train, x = x_knn, method="knn", tuneLength=20,
                          preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_knn_fullRFE$resample)-1))){  # Get the mean result
  knn_results_rfe <- cbind(knn_results_rfe, mean(model_knn_fullRFE$resample[, i]))
}
colnames(knn_results_rfe) <- names((model_knn_fullRFE$resample)[1:(length(model_knn_fullRFE$resample)-1)])
knn_results_rfe  # Show the result metrics

# The following part will be applied on the Grubbs' Outliers-removed Train Set :
set.seed(7)
knn_results_rfe_grubbs <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats)
model_knn_grubbsRFE = train(y = outGrubbs_train$Label, x = x_knn_grubbs, method="knn", tuneLength=20,
                            preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_knn_grubbsRFE$resample)-1))){  # Get the mean result
  knn_results_rfe_grubbs <- cbind(knn_results_rfe_grubbs, mean(model_knn_grubbsRFE$resample[, i]))
}
colnames(knn_results_rfe_grubbs) <- names((model_knn_grubbsRFE$resample)[1:(length(model_knn_grubbsRFE$resample)-1)])
knn_results_rfe_grubbs  # Show the result metrics

# # The following part will be applied on the Rosner' Outliers-removed Train Set :
set.seed(7)
knn_results_rfe_rosner <- NULL
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats)
model_knn_rosnerRFE = train(y = outRosner_train$Label, x = x_knn_rosner, method="knn", tuneLength=20,
                            preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_knn_rosnerRFE$resample)-1))){  # Get the mean result
  knn_results_rfe_rosner <- cbind(knn_results_rfe_rosner, mean(model_knn_rosnerRFE$resample[, i]))
}
colnames(knn_results_rfe_rosner) <- names((model_knn_rosnerRFE$resample)[1:(length(model_knn_rosnerRFE$resample)-1)])
knn_results_rfe_rosner  # Show the result metrics

#
# # -----------  KNN on Full Train Data :  -----------
set.seed(7)
knn_results_f <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_knn_full = train(y = y_train, x = x_train_data, method="knn", tuneLength=20,
                       preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_knn_full$resample)-1))){  # Get the mean result
  knn_results_f <- cbind(knn_results_f, mean(model_knn_full$resample[, i]))
}
colnames(knn_results_f) <- names((model_knn_full$resample)[1:(length(model_knn_full$resample)-1)])
knn_results_f  # Show the result metrics

# -----------  KNN on Outliers-removed Train Data :  -----------

# ------- Grubbs Outliers-removed Train set : -------
set.seed(7)
knn_results_grubbs <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_knn_grubbs = train(y = outGrubbs_train$Label, x = outGrubbs_train[, -length(train_data[,])], method="knn", tuneLength=20,
                         preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_knn_grubbs$resample)-1))){
  knn_results_grubbs <- cbind(knn_results_grubbs, mean(model_knn_grubbs$resample[, i]))
}
colnames(knn_results_grubbs) <- names((model_knn_grubbs$resample)[1:(length(model_knn_grubbs$resample)-1)])
knn_results_grubbs  # Show the result metrics

# ------- Rosner Outliers-removed Train set : -------
set.seed(7)
knn_results_rosner <- NULL  # Vector to store all metrics results
yt <- (outRosner_train['Label'])  # Label vector
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_knn_rosner = train(y = factor(yt[,]), x = outRosner_train[, -length(train_data[,])], method="knn", tuneLength=20,
                         preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_knn_rosner$resample)-1))){
  knn_results_rosner <- cbind(knn_results_rosner, mean(model_knn_rosner$resample[, i]))
}
colnames(knn_results_rosner) <- names((model_knn_rosner$resample)[1:(length(model_knn_rosner$resample)-1)])
knn_results_rosner  # Show the result metrics

# -----------  KNN on Correlation-removed Train Data :  -----------
# Full Train Data (with outliers)
set.seed(7)
knn_results_r <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_knn_noCorr = train(y = y_train, x = x_reduced_train, method="knn", tuneLength=20,
                         preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_knn_noCorr$resample)-1))){  # Get the mean result
  knn_results_r <- cbind(knn_results_r, mean(model_knn_noCorr$resample[, i]))
}
colnames(knn_results_r) <- names((model_knn_noCorr$resample)[1:(length(model_knn_noCorr$resample)-1)])
knn_results_r  # Show the result metrics

# Grubbs - Train Data (without outliers and no correlation)
set.seed(7)
knn_results_rGrubbs <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_knn_grubbsNoCorr = train(y = outGrubbs_train$Label, x = x_reduced_train_grubbs, method="knn", tuneLength=20,
                               preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_knn_grubbsNoCorr$resample)-1))){  # Get the mean result
  knn_results_rGrubbs <- cbind(knn_results_rGrubbs, mean(model_knn_grubbsNoCorr$resample[, i]))
}
colnames(knn_results_rGrubbs) <- names((model_knn_grubbsNoCorr$resample)[1:(length(model_knn_grubbsNoCorr$resample)-1)])
knn_results_rGrubbs  # Show the result metrics

# Rosner - Train Data (without outliers and no correlation)
set.seed(7)
knn_results_rRosner <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_knn_rosnerNoCorr = train(y = outRosner_train$Label, x = x_reduced_train_rosner, method="knn", tuneLength=20,
                               preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_knn_rosnerNoCorr$resample)-1))){  # Get the mean result
  knn_results_rRosner <- cbind(knn_results_rRosner, mean(model_knn_rosnerNoCorr$resample[, i]))
}
colnames(knn_results_rRosner) <- names((model_knn_rosnerNoCorr$resample)[1:(length(model_knn_rosnerNoCorr$resample)-1)])
knn_results_rRosner  # Show the result metrics


# -----------  KNN on PCA Train Data :  -----------

# The following part will apply PCA on the whole Train Set, with different thresholds :
set.seed(7)
knn_results_pca <- NULL  # Vector to store all metrics results
for(i in seq(70,95,5)){  # Looping over different thresholds for PCA
  control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                          preProcOptions = list(thresh = i/100), classProbs = TRUE, summaryFunction = metricStats)
  model_knn_fullPCA = train(y = y_train, x = x_train_data, method="knn", tuneLength=20,
                            preProcess=c("center", "scale", "pca"), trControl = control)
  knn_results_pca <- rbind(knn_results_pca, cbind(model_knn_fullPCA$results$pAUC, model_knn_fullPCA$results$MCC, model_knn_fullPCA$results$ROC, model_knn_fullPCA$results$Sens, model_knn_fullPCA$results$Spec, model_knn_fullPCA$results$Accuracy, model_knn_fullPCA$results$Kappa, model_knn_fullPCA$results$AUC, model_knn_fullPCA$results$Precision, model_knn_fullPCA$results$Recall, model_knn_fullPCA$results$F))  # Storing the metrics scores
  colnames(knn_results_pca) <- cbind('pAUC', 'MCC', 'ROC', 'Sens', 'Spec', 'Acc', 'Kappa', 'AUC', 'Precision', 'Recall', 'F') }
knn_results_pca <- knn_results_pca[which.max(knn_results_pca[,1]),]  # Keep the one with max pAUC amoung the different thresholds
knn_results_pca  # Show the result metrics


# The following part will apply PCA on the Correlation-removed Train Set, with different thresholds :
set.seed(7)
knn_results_rpca <- NULL  # Vector to store all metrics results
for(i in seq(70,95,5)){  # Looping over different thresholds for PCA
  control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                          preProcOptions = list(thresh = i/100), classProbs = TRUE, summaryFunction = metricStats)
  model_knn_rPCA = train(y = y_train, x = x_reduced_train, method="knn", tuneLength=20,
                         preProcess=c("center", "scale", "pca"), trControl = control)
  knn_results_rpca <- rbind(knn_results_rpca, cbind(model_knn_rPCA$results$pAUC, model_knn_rPCA$results$MCC, model_knn_rPCA$results$ROC, model_knn_rPCA$results$Sens, model_knn_rPCA$results$Spec, model_knn_rPCA$results$Accuracy, model_knn_rPCA$results$Kappa, model_knn_rPCA$results$AUC, model_knn_rPCA$results$Precision, model_knn_rPCA$results$Recall, model_knn_rPCA$results$F))
  colnames(knn_results_rpca) <- cbind('pAUC', 'MCC', 'ROC', 'Sens', 'Spec', 'Acc', 'Kappa', 'AUC', 'Precision', 'Recall', 'F') }
knn_results_rpca <- knn_results_rpca[which.max(knn_results_rpca[,1]),]  # Keep the one with max pAUC amoung the different thresholds
knn_results_rpca  # Show the result metrics

# The following part will apply PCA on the Grubbs Outliers-removed Train Set, with different thresholds :
set.seed(7)
knn_results_rpca_grubbs <- NULL  # Vector to store all metrics results
for(i in seq(70,95,5)){  # Looping over different thresholds for PCA
  control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                          preProcOptions = list(thresh = i/100), classProbs = TRUE, summaryFunction = metricStats)
  model_knn_grubbsPCA = train(y = outGrubbs_train$Label, x = x_reduced_train_grubbs, method="knn", tuneLength=20,
                              preProcess=c("center", "scale", "pca"), trControl = control)
  knn_results_rpca_grubbs <- rbind(knn_results_rpca_grubbs, cbind(model_knn_grubbsPCA$results$pAUC, model_knn_grubbsPCA$results$MCC, model_knn_grubbsPCA$results$ROC, model_knn_grubbsPCA$results$Sens, model_knn_grubbsPCA$results$Spec, model_knn_grubbsPCA$results$Accuracy, model_knn_grubbsPCA$results$Kappa, model_knn_grubbsPCA$results$AUC, model_knn_grubbsPCA$results$Precision, model_knn_grubbsPCA$results$Recall, model_knn_grubbsPCA$results$F))
  colnames(knn_results_rpca_grubbs) <- cbind('pAUC', 'MCC', 'ROC', 'Sens', 'Spec', 'Acc', 'Kappa', 'AUC', 'Precision', 'Recall', 'F') }
knn_results_rpca_grubbs <- knn_results_rpca_grubbs[which.max(knn_results_rpca_grubbs[,1]),]  # Keep the one with max pAUC amoung the different thresholds
knn_results_rpca_grubbs  # Show the result metrics

# The following part will apply PCA on the Rosner Outliers-removed Train Set, with different thresholds :
set.seed(7)
knn_results_rpca_rosner <- NULL  # Vector to store all metrics results
for(i in seq(70,95,5)){  # Looping over different thresholds for PCA
  control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                          preProcOptions = list(thresh = i/100), classProbs = TRUE, summaryFunction = metricStats)
  model_knn_rosnerPCA = train(y = outRosner_train$Label, x = x_reduced_train_rosner, method="knn", tuneLength=20,
                              preProcess=c("center", "scale", "pca"), trControl = control)
  knn_results_rpca_rosner <- rbind(knn_results_rpca_rosner, cbind(model_knn_rosnerPCA$results$pAUC, model_knn_rosnerPCA$results$MCC, model_knn_rosnerPCA$results$ROC, model_knn_rosnerPCA$results$Sens, model_knn_rosnerPCA$results$Spec, model_knn_rosnerPCA$results$Accuracy, model_knn_rosnerPCA$results$Kappa, model_knn_rosnerPCA$results$AUC, model_knn_rosnerPCA$results$Precision, model_knn_rosnerPCA$results$Recall, model_knn_rosnerPCA$results$F))
  colnames(knn_results_rpca_rosner) <- cbind('pAUC', 'MCC', 'ROC', 'Sens', 'Spec', 'Acc', 'Kappa', 'AUC', 'Precision', 'Recall', 'F') }
knn_results_rpca_rosner <- knn_results_rpca_rosner[which.max(knn_results_rpca_rosner[,1]),]  # Keep the one with max pAUC amoung the different thresholds
knn_results_rpca_rosner  # Show the result metrics


# -------  Summary of the different results :  -------

res_knn <- rbind(knn_results_f, knn_results_grubbs, knn_results_rosner,
                 knn_results_r, knn_results_rGrubbs, knn_results_rRosner,
                 knn_results_rfe, knn_results_rfe_grubbs, knn_results_rfe_rosner)
rownames(res_knn) <- cbind('Full', 'Grubbs', 'Rosner', 'FullNoCorr', 'GrubbsNoCorr', 'rosnerNoCorr',
                           'FullRFE', 'GrubbsRFE', 'RosnerRFE')
res_knn <- res_knn[, c(11,10,1,2,3,4,5,6,7,8,9)]  # Reorder the columns to compare
resPca_knn <- rbind(knn_results_pca, knn_results_rpca, knn_results_rpca_grubbs, knn_results_rpca_rosner)
rownames(resPca_knn) <- cbind('FullPCA', 'ReducedPCA', 'GrubbsPCA', 'RosnerPCA')
# Show results :
scores_knn <- rbind(res_knn, resPca_knn)
scores_knn


# # # ## ----  FINAL RESULTS - KNN :  ----
#              pAUC   MCC    ROC   Sens  Spec  Accuracy Kappa AUC   Precision Recall F
# Full         0.853  0.716  0.902 0.848 0.859 0.854    0.707 0.726 0.865     0.848  0.851
# Grubbs       0.858  0.722  0.885 0.859 0.856 0.856    0.711 0.601 0.843     0.859  0.843
# Rosner       0.854  0.719  0.910 0.863 0.846 0.854    0.709 0.731 0.857     0.863  0.852
# FullNoCorr   0.847  0.703  0.905 0.831 0.863 0.847    0.694 0.723 0.860     0.831  0.840
# GrubbsNoCorr 0.849  0.705  0.901 0.841 0.856 0.847    0.692 0.671 0.836     0.841  0.829
# rosnerNoCorr 0.863  0.734  0.914 0.855 0.871 0.863    0.726 0.723 0.868     0.855  0.855
# FullRFE      0.884  0.776  0.926 0.876 0.891 0.884    0.768 0.509 0.898     0.876  0.882
# GrubbsRFE    0.890  0.786  0.935 0.903 0.876 0.887    0.774 0.595 0.872     0.903  0.879
# RosnerRFE    0.895  0.801  0.952 0.864 0.925 0.896    0.791 0.472 0.920     0.864  0.885
# FullPCA      0.853  0.716  0.897 0.839 0.868 0.853    0.706 0.712 0.871     0.839  0.848
# ReducedPCA   0.844  0.699  0.897 0.838 0.850 0.844    0.688 0.731 0.851     0.838  0.837
# GrubbsPCA    0.853  0.713  0.888 0.866 0.840 0.850    0.700 0.619 0.829     0.866  0.838
# RosnerPCA    0.871  0.754  0.919 0.855 0.887 0.872    0.743 0.709 0.884     0.855  0.861

# # Best KNN :
# RosnerRFE    0.895  0.801  0.952 0.864 0.925 0.896    0.791 0.472 0.920     0.864  0.885

bestModel_KNN <- model_knn_rosnerRFE

## -----------------------------------------------------------------------------



## -----------------------------------------------------------------------------
## --------------------------  LINEAR MODEL (GLM)  -----------------------------
## -----------------------------------------------------------------------------

# ---  Define the RFE function for GLM  ---
ctrl$functions <- lrFuncs
ctrl$functions$summary <- metricStats
set.seed(7)
# Run the RFE Algorithm, whose performance is based on AUC metric, for different sizes (number of possible features) :
glmRFE <- rfe(x=x_train_data, y=y_train, metric = "pAUC",
              sizes = c(4,8,10,12,14,16,18,20,22,24,50,100,200,300,400),
              preProc = c("center", "scale"), rfeControl = ctrl)
glmRFE  # Show the results
plot(glmRFE, type=c("g", "o"))  # Plot the AUC measures for different number of variables
glmPredict = predictors(glmRFE)  # Get the retained predictors
x_glm = subset(train_data, select = glmPredict)  # Keep only the RFE features for the Train Set

##  --->  We do the same steps for the Outliers removed Data (Both for Grubbs and Rosner) :
set.seed(7)  # Grubbs - Removed outliers
glmRFE_grubbs <- rfe(x=x_train_data_grubbs, y=factor(outGrubbs_train$Label),
                     sizes = c(4,8,10,12,14,16,18,20,22,24,50,100,200,300,400),
                     metric = "pAUC", preProc = c("center", "scale"), rfeControl = ctrl)
glmPredict_grubbs = predictors(glmRFE_grubbs)
# plot(glmRFE_grubbs, type=c("g", "o"))  # Plot the AUC measures for different number of variables
x_glm_grubbs = subset(outGrubbs_train, select = glmPredict_grubbs)

set.seed(7)  # Rosner - Removed outliers
glmRFE_rosner <- rfe(x=x_train_data_rosner, y=factor(outRosner_train$Label),
                     sizes = c(4,8,10,12,14,16,18,20,22,24,50,100,200,300,400), 
                     metric = "pAUC", preProc = c("center", "scale"), rfeControl = ctrl)
glmPredict_rosner = predictors(glmRFE_rosner)
# plot(glmRFE_rosner, type=c("g", "o"))  # Plot the AUC measures for different number of variables
x_glm_rosner = subset(outRosner_train, select = glmPredict_rosner)


## -----------------------------------------------------------------------------
##  ---------------   Comparing different GLM Training   -----------------------

# -----------  GLM on RFE Train Data :  -----------
# The following part will be applied on the whole Train Set :
set.seed(7)
glm_results_rfe <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_glm_fullRFE = train(y = y_train, x = x_glm, method="glm",
                          preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_glm_fullRFE$resample)-1))){  # Get the mean result
  glm_results_rfe <- cbind(glm_results_rfe, mean(model_glm_fullRFE$resample[, i]))
}
colnames(glm_results_rfe) <- names((model_glm_fullRFE$resample)[1:(length(model_glm_fullRFE$resample)-1)])
glm_results_rfe  # Show the result metrics

# The following part will be applied on the Grubbs' Outliers-removed Train Set :
set.seed(7)
glm_results_rfe_grubbs <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_glm_grubbsRFE = train(y = outGrubbs_train$Label, x = x_glm_grubbs, method="glm",
                            preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_glm_grubbsRFE$resample)-1))){
  glm_results_rfe_grubbs <- cbind(glm_results_rfe_grubbs, mean(model_glm_grubbsRFE$resample[, i]))
}
colnames(glm_results_rfe_grubbs) <- names((model_glm_grubbsRFE$resample)[1:(length(model_glm_grubbsRFE$resample)-1)])
glm_results_rfe_grubbs  # Show the result metrics

# The following part will be applied on the Rosner' Outliers-removed Train Set :
set.seed(7)
glm_results_rfe_rosner <- NULL
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_glm_rosnerRFE = train(y = outRosner_train$Label, x = x_glm_rosner, method="glm",
                            preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_glm_rosnerRFE$resample)-1))){
  glm_results_rfe_rosner <- cbind(glm_results_rfe_rosner, mean(model_glm_rosnerRFE$resample[, i]))
}
colnames(glm_results_rfe_rosner) <- names((model_glm_rosnerRFE$resample)[1:(length(model_glm_rosnerRFE$resample)-1)])
glm_results_rfe_rosner  # Show the result metrics


# -----------  glm on Full Train Data :  -----------
set.seed(7)
glm_results_f <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_glm_full = train(y = y_train, x = x_train_data, method="glm",
                       preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_glm_full$resample)-1))){
  glm_results_f <- cbind(glm_results_f, mean(model_glm_full$resample[, i]))
}
colnames(glm_results_f) <- names((model_glm_full$resample)[1:(length(model_glm_full$resample)-1)])
glm_results_f  # Show the result metrics

# -----------  glm on Outliers-removed Train Data :  -----------

# ------- Grubbs Outliers-removed Train set : -------
set.seed(7)
glm_results_grubbs <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_glm_grubbs = train(y = outGrubbs_train$Label, x = outGrubbs_train[, -length(train_data[,])], method="glm",
                         preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_glm_grubbs$resample)-1))){
  glm_results_grubbs <- cbind(glm_results_grubbs, mean(model_glm_grubbs$resample[, i]))
}
colnames(glm_results_grubbs) <- names((model_glm_grubbs$resample)[1:(length(model_glm_grubbs$resample)-1)])
glm_results_grubbs  # Show the result metrics

# ------- Rosner Outliers-removed Train set : -------
set.seed(7)
glm_results_rosner <- NULL  # Vector to store all metrics results
yt <- (outRosner_train['Label'])  # Label vector
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_glm_rosner = train(y = factor(yt[,]), x = outRosner_train[, -length(train_data[,])], method="glm",
                         preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_glm_rosner$resample)-1))){
  glm_results_rosner <- cbind(glm_results_rosner, mean(model_glm_rosner$resample[, i]))
}
colnames(glm_results_rosner) <- names((model_glm_rosner$resample)[1:(length(model_glm_rosner$resample)-1)])
glm_results_rosner  # Show the result metrics


# -----------  glm on Correlation-removed Train Data :  -----------
# Full Train Data (with outliers and no correlation)
set.seed(7)
glm_results_r <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_glm_noCorr = train(y = y_train, x = x_reduced_train, method="glm",
                         preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_glm_noCorr$resample)-1))){
  glm_results_r <- cbind(glm_results_r, mean(model_glm_noCorr$resample[, i]))
}
colnames(glm_results_r) <- names((model_glm_noCorr$resample)[1:(length(model_glm_noCorr$resample)-1)])
glm_results_r  # Show the result metrics

# Grubbs - Train Data (without outliers and no correlation)
set.seed(7)
glm_results_rGrubbs <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_glm_grubbsNoCorr = train(y = outGrubbs_train$Label, x = x_reduced_train_grubbs, method="glm",
                               preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_glm_grubbsNoCorr$resample)-1))){
  glm_results_rGrubbs <- cbind(glm_results_rGrubbs, mean(model_glm_grubbsNoCorr$resample[, i]))
}
colnames(glm_results_rGrubbs) <- names((model_glm_grubbsNoCorr$resample)[1:(length(model_glm_grubbsNoCorr$resample)-1)])
glm_results_rGrubbs  # Show the result metrics

# Rosner - Train Data (without outliers and no correlation)
set.seed(7)
glm_results_rRosner <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_glm_rosnerNoCorr = train(y = outRosner_train$Label, x = x_reduced_train_rosner, method="glm",
                               preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_glm_rosnerNoCorr$resample)-1))){
  glm_results_rRosner <- cbind(glm_results_rRosner, mean(model_glm_rosnerNoCorr$resample[, i]))
}
colnames(glm_results_rRosner) <- names((model_glm_rosnerNoCorr$resample)[1:(length(model_glm_rosnerNoCorr$resample)-1)])
glm_results_rRosner  # Show the result metrics


# -----------  glm on PCA Train Data :  -----------

# The following part will apply PCA on the whole Train Set, with different thresholds :
set.seed(7)
glm_results_pca <- NULL  # Vector to store all metrics results
for(i in seq(70,95,5)){  # Looping over different thresholds for PCA
  control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                          preProcOptions = list(thresh = i/100), classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
  model_glm_fullPCA = train(y = y_train, x = x_train_data, method="glm",
                            preProcess=c("center", "scale", "pca"), trControl = control)
  glm_results_pca <- rbind(glm_results_pca, cbind(model_glm_fullPCA$results$pAUC, model_glm_fullPCA$results$MCC, model_glm_fullPCA$results$ROC, model_glm_fullPCA$results$Sens, model_glm_fullPCA$results$Spec, model_glm_fullPCA$results$Accuracy, model_glm_fullPCA$results$Kappa, model_glm_fullPCA$results$AUC, model_glm_fullPCA$results$Precision, model_glm_fullPCA$results$Recall, model_glm_fullPCA$results$F))
  colnames(glm_results_pca) <- cbind('pAUC', 'MCC', 'ROC', 'Sens', 'Spec', 'Acc', 'Kappa', 'AUC', 'Precision', 'Recall', 'F') }
glm_results_pca <- glm_results_pca[which.max(glm_results_pca[,1]),]  # Keep the one with max pAUC amoung the different thresholds
glm_results_pca  # Show the result metrics


# The following part will apply PCA on the Correlation-removed Train Set, with different thresholds :
set.seed(7)
glm_results_rpca <- NULL  # Vector to store all metrics results
for(i in seq(70,95,5)){  # Looping over different thresholds for PCA
  control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                          preProcOptions = list(thresh = i/100), classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
  model_glm_rPCA = train(y = y_train, x = x_reduced_train, method="glm",
                         preProcess=c("center", "scale", "pca"), trControl = control)
  glm_results_rpca <- rbind(glm_results_rpca, cbind(model_glm_rPCA$results$pAUC, model_glm_rPCA$results$MCC, model_glm_rPCA$results$ROC, model_glm_rPCA$results$Sens, model_glm_rPCA$results$Spec, model_glm_rPCA$results$Accuracy, model_glm_rPCA$results$Kappa, model_glm_rPCA$results$AUC, model_glm_rPCA$results$Precision, model_glm_rPCA$results$Recall, model_glm_rPCA$results$F))
  colnames(glm_results_rpca) <- cbind('pAUC', 'MCC', 'ROC', 'Sens', 'Spec', 'Acc', 'Kappa', 'AUC', 'Precision', 'Recall', 'F') }
glm_results_rpca <- glm_results_rpca[which.max(glm_results_rpca[,1]),]  # Keep the one with max pAUC amoung the different thresholds
glm_results_rpca  # Show the result metrics

# The following part will apply PCA on the Grubbs Outliers-removed Train Set, with different thresholds :
set.seed(7)
glm_results_rpca_grubbs <- NULL  # Vector to store all metrics results
for(i in seq(70,95,5)){  # Looping over different thresholds for PCA
  control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                          preProcOptions = list(thresh = i/100), classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
  model_glm_grubbsPCA = train(y = outGrubbs_train$Label, x = x_reduced_train_grubbs, method="glm",
                              preProcess=c("center", "scale", "pca"), trControl = control)
  glm_results_rpca_grubbs <- rbind(glm_results_rpca_grubbs, cbind(model_glm_grubbsPCA$results$pAUC, model_glm_grubbsPCA$results$MCC, model_glm_grubbsPCA$results$ROC, model_glm_grubbsPCA$results$Sens, model_glm_grubbsPCA$results$Spec, model_glm_grubbsPCA$results$Accuracy, model_glm_grubbsPCA$results$Kappa, model_glm_grubbsPCA$results$AUC, model_glm_grubbsPCA$results$Precision, model_glm_grubbsPCA$results$Recall, model_glm_grubbsPCA$results$F))
  colnames(glm_results_rpca_grubbs) <- cbind('pAUC', 'MCC', 'ROC', 'Sens', 'Spec', 'Acc', 'Kappa', 'AUC', 'Precision', 'Recall', 'F') }
glm_results_rpca_grubbs <- glm_results_rpca_grubbs[which.max(glm_results_rpca_grubbs[,1]),]  # Keep the one with max pAUC amoung the different thresholds
glm_results_rpca_grubbs  # Show the result metrics

# The following part will apply PCA on the Rosner Outliers-removed Train Set, with different thresholds :
set.seed(7)
glm_results_rpca_rosner <- NULL  # Vector to store all metrics results
for(i in seq(70,95,5)){  # Looping over different thresholds for PCA
  control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                          preProcOptions = list(thresh = i/100), classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
  model_glm_rosnerPCA = train(y = outRosner_train$Label, x = x_reduced_train_rosner, method="glm",
                              preProcess=c("center", "scale", "pca"), trControl = control)
  glm_results_rpca_rosner <- rbind(glm_results_rpca_rosner, cbind(model_glm_rosnerPCA$results$pAUC, model_glm_rosnerPCA$results$MCC, model_glm_rosnerPCA$results$ROC, model_glm_rosnerPCA$results$Sens, model_glm_rosnerPCA$results$Spec, model_glm_rosnerPCA$results$Accuracy, model_glm_rosnerPCA$results$Kappa, model_glm_rosnerPCA$results$AUC, model_glm_rosnerPCA$results$Precision, model_glm_rosnerPCA$results$Recall, model_glm_rosnerPCA$results$F))
  colnames(glm_results_rpca_rosner) <- cbind('pAUC', 'MCC', 'ROC', 'Sens', 'Spec', 'Acc', 'Kappa', 'AUC', 'Precision', 'Recall', 'F') }
glm_results_rpca_rosner <- glm_results_rpca_rosner[which.max(glm_results_rpca_rosner[,1]),]  # Keep the one with max pAUC amoung the different thresholds
glm_results_rpca_rosner  # Show the result metrics


# -----  Summary of the different results :
res_glm <- rbind(glm_results_f, glm_results_grubbs, glm_results_rosner,
                 glm_results_r, glm_results_rGrubbs, glm_results_rRosner,
                 glm_results_rfe, glm_results_rfe_grubbs, glm_results_rfe_rosner)
rownames(res_glm) <- cbind('Full', 'Grubbs', 'Rosner', 'FullNoCorr', 'GrubbsNoCorr', 'rosnerNoCorr',
                           'FullRFE', 'GrubbsRFE', 'RosnerRFE')
res_glm <- res_glm[, c(11,10,1,2,3,4,5,6,7,8,9)]  # Reorder the columns to compare
resPca_glm <- rbind(glm_results_pca, glm_results_rpca, glm_results_rpca_grubbs, glm_results_rpca_rosner)
rownames(resPca_glm) <- cbind('FullPCA', 'ReducedPCA', 'GrubbsPCA', 'RosnerPCA')
# Show results :
scores_glm <- rbind(res_glm, resPca_glm)
scores_glm


# # ## ----  FINAL RESULTS - GLM :  ----
#              pAUC   MCC     ROC   Sens  Spec  Accuracy Kappa   AUC   Precision Recall F
# Full         0.603  0.09833 0.545 0.572 0.526 0.548    0.09742 0.494 0.543     0.572  0.549
# Grubbs       0.597  0.00969 0.524 0.463 0.548 0.509    0.00901 0.438 0.456     0.463  NaN
# Rosner       0.602  0.10968 0.548 0.546 0.558 0.554    0.10458 0.470 0.525     0.546  0.527
# FullNoCorr   0.583  0.00864 0.533 0.466 0.542 0.504    0.00845 0.499 0.505     0.466  0.475
# GrubbsNoCorr 0.552 -0.00732 0.505 0.512 0.486 0.500   -0.00202 0.425 0.447     0.512  NaN
# rosnerNoCorr 0.611  0.09508 0.548 0.549 0.542 0.546    0.08924 0.477 0.512     0.549  0.524
# FullRFE      0.825  0.65778 0.894 0.848 0.802 0.825    0.64998 0.770 0.811     0.848  0.826
# GrubbsRFE    0.727  0.46894 0.823 0.714 0.740 0.727    0.45234 0.648 0.718     0.714  0.701
# RosnerRFE    0.826  0.66076 0.902 0.848 0.804 0.824    0.64879 0.766 0.807     0.848  0.819
# FullPCA      0.898  0.80234 0.958 0.922 0.874 0.898    0.79620 0.831 0.886     0.922  0.901
# ReducedPCA   0.880  0.77127 0.939 0.930 0.831 0.880    0.76059 0.812 0.854     0.930  0.886
# GrubbsPCA    0.845  0.70025 0.908 0.886 0.805 0.839    0.67995 0.705 0.804     0.886  0.832
# RosnerPCA    0.891  0.79016 0.954 0.907 0.875 0.890    0.77925 0.802 0.875     0.907  0.884


# Best GLM :
# FullPCA      0.898  0.80234 0.958 0.922 0.874 0.898    0.79620 0.831 0.886     0.922  0.901
bestModel_GLM <- model_glm_fullPCA

# ---------------------------------------------------------------------------------------------------


## -----------------------------------------------------------------------------
## --------------------------  NAIVE BAYES (NB)  -----------------------------
## -----------------------------------------------------------------------------

# ---  Define the RFE function for NB  ---
ctrl$functions <- nbFuncs
ctrl$functions$summary <- metricStats
set.seed(7)
# Run the RFE Algorithm, whose performance is based on AUC metric, for different sizes (number of possible features) :
nbRFE <- rfe(x=x_train_data, y=y_train, metric = "pAUC",
             sizes = c(4,8,10,12,14,16,18,20,22,24,50,100,200,300,400),
             preProc = c("center", "scale"), rfeControl = ctrl)
nbRFE  # Show the results
plot(nbRFE, type=c("g", "o"))  # Plot the AUC measures for different number of variables
nbPredict = predictors(nbRFE)  # Get the retained predictors
x_nb = subset(train_data, select = nbPredict)  # Keep only the RFE features for the Train Set

##  --->  We do the same steps for the Outliers removed Data (Both for Grubbs and Rosner) :
set.seed(7)  # Grubbs - Removed outliers
nbRFE_grubbs <- rfe(x=x_train_data_grubbs, y=factor(outGrubbs_train$Label),
                    sizes = c(4,8,10,12,14,16,18,20,22,24,50,100,200,300,400),
                    metric = "pAUC", preProc = c("center", "scale"), rfeControl = ctrl)
nbPredict_grubbs = predictors(nbRFE_grubbs)
# plot(nbRFE_grubbs, type=c("g", "o"))  # Plot the AUC measures for different number of variables
x_nb_grubbs = subset(outGrubbs_train, select = nbPredict_grubbs)

set.seed(7)  # Rosner - Removed outliers
nbRFE_rosner <- rfe(x=x_train_data_rosner, y=factor(outRosner_train$Label),
                    sizes = c(4,8,10,12,14,16,18,20,22,24,50,100,200,300,400),
                    metric = "pAUC", preProc = c("center", "scale"), rfeControl = ctrl)
nbPredict_rosner = predictors(nbRFE_rosner)
# plot(nbRFE_rosner, type=c("g", "o"))  # Plot the AUC measures for different number of variables
x_nb_rosner = subset(outRosner_train, select = nbPredict_rosner)


## -----------------------------------------------------------------------------
##  ---------------   Comparing different NB Training   -----------------------

# -----------  NB on RFE Train Data :  -----------
# The following part will be applied on the whole Train Set :
set.seed(7)
nb_results_rfe <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_nb_fullRFE = train(y = y_train, x = x_nb, method="nb",
                         preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_nb_fullRFE$resample)-1))){  # Get the mean result
  nb_results_rfe <- cbind(nb_results_rfe, mean(model_nb_fullRFE$resample[, i]))
}
colnames(nb_results_rfe) <- names((model_nb_fullRFE$resample)[1:(length(model_nb_fullRFE$resample)-1)])
nb_results_rfe  # Show the result metrics


# The following part will be applied on the Grubbs' Outliers-removed Train Set :
set.seed(7)
nb_results_rfe_grubbs <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_nb_grubbsRFE = train(y = outGrubbs_train$Label, x = x_nb_grubbs, method="nb",
                           preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_nb_grubbsRFE$resample)-1))){
  nb_results_rfe_grubbs <- cbind(nb_results_rfe_grubbs, mean(model_nb_grubbsRFE$resample[, i]))
}
colnames(nb_results_rfe_grubbs) <- names((model_nb_grubbsRFE$resample)[1:(length(model_nb_grubbsRFE$resample)-1)])
nb_results_rfe_grubbs  # Show the result metrics

# The following part will be applied on the Rosner' Outliers-removed Train Set :
set.seed(7)
nb_results_rfe_rosner <- NULL
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_nb_rosnerRFE = train(y = outRosner_train$Label, x = x_nb_rosner, method="nb",
                           preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_nb_rosnerRFE$resample)-1))){
  nb_results_rfe_rosner <- cbind(nb_results_rfe_rosner, mean(model_nb_rosnerRFE$resample[, i]))
}
colnames(nb_results_rfe_rosner) <- names((model_nb_rosnerRFE$resample)[1:(length(model_nb_rosnerRFE$resample)-1)])
nb_results_rfe_rosner  # Show the result metrics


# -----------  NB on Full Train Data :  -----------
set.seed(7)
nb_results_f <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_nb_full = train(y = y_train, x = x_train_data, method="nb",
                      preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_nb_full$resample)-1))){
  nb_results_f <- cbind(nb_results_f, mean(model_nb_full$resample[, i]))
}
colnames(nb_results_f) <- names((model_nb_full$resample)[1:(length(model_nb_full$resample)-1)])
nb_results_f  # Show the result metrics

# -----------  NB on Outliers-removed Train Data :  -----------
# ------- Rosner Outliers-removed Train set : -------
set.seed(7)
nb_results_rosner <- NULL  # Vector to store all metrics results
yt <- (outRosner_train['Label'])  # Label vector
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_nb_rosner = train(y = factor(yt[,]), x = outRosner_train[, -length(train_data[,])], method="nb",
                        preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_nb_rosner$resample)-1))){
  nb_results_rosner <- cbind(nb_results_rosner, mean(model_nb_rosner$resample[, i]))
}
colnames(nb_results_rosner) <- names((model_nb_rosner$resample)[1:(length(model_nb_rosner$resample)-1)])
nb_results_rosner  # Show the result metrics

# ------- Grubbs Outliers-removed Train set : -------
set.seed(7)
nb_results_grubbs <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_nb_grubbs = train(y = outGrubbs_train$Label, x = outGrubbs_train[, -length(train_data[,])], method="nb",
                        preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_nb_grubbs$resample)-1))){
  nb_results_grubbs <- cbind(nb_results_grubbs, mean(model_nb_grubbs$resample[, i]))
}
colnames(nb_results_grubbs) <- names((model_nb_grubbs$resample)[1:(length(model_nb_grubbs$resample)-1)])
nb_results_grubbs  # Show the result metrics


# -----------  NB on Correlation-removed Train Data :  -----------
# Full Train Data (with outliers and no correlation)
set.seed(7)
nb_results_r <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_nb_noCorr = train(y = y_train, x = x_reduced_train, method="nb",
                        preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_nb_noCorr$resample)-1))){
  nb_results_r <- cbind(nb_results_r, mean(model_nb_noCorr$resample[, i]))
}
colnames(nb_results_r) <- names((model_nb_noCorr$resample)[1:(length(model_nb_noCorr$resample)-1)])
nb_results_r  # Show the result metrics

# Grubbs - Train Data (without outliers and no correlation)
set.seed(7)
nb_results_rGrubbs <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_nb_grubbsNoCorr = train(y = outGrubbs_train$Label, x = x_reduced_train_grubbs, method="nb",
                              preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_nb_grubbsNoCorr$resample)-1))){
  nb_results_rGrubbs <- cbind(nb_results_rGrubbs, mean(model_nb_grubbsNoCorr$resample[, i]))
}
colnames(nb_results_rGrubbs) <- names((model_nb_grubbsNoCorr$resample)[1:(length(model_nb_grubbsNoCorr$resample)-1)])
nb_results_rGrubbs  # Show the result metrics

# Rosner - Train Data (without outliers and no correlation)
set.seed(7)
nb_results_rRosner <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_nb_rosnerNoCorr = train(y = outRosner_train$Label, x = x_reduced_train_rosner, method="nb",
                              preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_nb_rosnerNoCorr$resample)-1))){
  nb_results_rRosner <- cbind(nb_results_rRosner, mean(model_nb_rosnerNoCorr$resample[, i]))
}
colnames(nb_results_rRosner) <- names((model_nb_rosnerNoCorr$resample)[1:(length(model_nb_rosnerNoCorr$resample)-1)])
nb_results_rRosner  # Show the result metrics


# -----------  NB on PCA Train Data :  -----------

# The following part will apply PCA on the whole Train Set, with different thresholds :
set.seed(7)
nb_results_pca <- NULL  # Vector to store all metrics results
for(i in seq(70,95,5)){  # Looping over different thresholds for PCA
  control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                          preProcOptions = list(thresh = i/100), classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
  model_nb_fullPCA = train(y = y_train, x = x_train_data, method="nb",
                           preProcess=c("center", "scale", "pca"), trControl = control)
  nb_results_pca <- rbind(nb_results_pca, cbind(model_nb_fullPCA$results$pAUC, model_nb_fullPCA$results$MCC, model_nb_fullPCA$results$ROC, model_nb_fullPCA$results$Sens, model_nb_fullPCA$results$Spec, model_nb_fullPCA$results$Accuracy, model_nb_fullPCA$results$Kappa, model_nb_fullPCA$results$AUC, model_nb_fullPCA$results$Precision, model_nb_fullPCA$results$Recall, model_nb_fullPCA$results$F))
  colnames(nb_results_pca) <- cbind('pAUC', 'MCC', 'ROC', 'Sens', 'Spec', 'Acc', 'Kappa', 'AUC', 'Precision', 'Recall', 'F') }
nb_results_pca <- nb_results_pca[which.max(nb_results_pca[,1]),]  # Keep the one with max pAUC amoung the different thresholds
nb_results_pca  # Show the result metrics


# The following part will apply PCA on the Correlation-removed Train Set, with different thresholds :
set.seed(7)
nb_results_rpca <- NULL  # Vector to store all metrics results
for(i in seq(70,95,5)){  # Looping over different thresholds for PCA
  control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                          preProcOptions = list(thresh = i/100), classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
  model_nb_rPCA = train(y = y_train, x = x_reduced_train, method="nb",
                        preProcess=c("center", "scale", "pca"), trControl = control)
  nb_results_rpca <- rbind(nb_results_rpca, cbind(model_nb_rPCA$results$pAUC, model_nb_rPCA$results$MCC, model_nb_rPCA$results$ROC, model_nb_rPCA$results$Sens, model_nb_rPCA$results$Spec, model_nb_rPCA$results$Accuracy, model_nb_rPCA$results$Kappa, model_nb_rPCA$results$AUC, model_nb_rPCA$results$Precision, model_nb_rPCA$results$Recall, model_nb_rPCA$results$F))
  colnames(nb_results_rpca) <- cbind('pAUC', 'MCC', 'ROC', 'Sens', 'Spec', 'Acc', 'Kappa', 'AUC', 'Precision', 'Recall', 'F') }
nb_results_rpca <- nb_results_rpca[which.max(nb_results_rpca[,1]),]  # Keep the one with max pAUC amoung the different thresholds
nb_results_rpca  # Show the result metrics

# The following part will apply PCA on the Grubbs Outliers-removed Train Set, with different thresholds :
set.seed(7)
nb_results_rpca_grubbs <- NULL  # Vector to store all metrics results
for(i in seq(70,95,5)){  # Looping over different thresholds for PCA
  control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                          preProcOptions = list(thresh = i/100), classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
  model_nb_grubbsPCA = train(y = outGrubbs_train$Label, x = x_reduced_train_grubbs, method="nb",
                             preProcess=c("center", "scale", "pca"), trControl = control)
  nb_results_rpca_grubbs <- rbind(nb_results_rpca_grubbs, cbind(model_nb_grubbsPCA$results$pAUC, model_nb_grubbsPCA$results$MCC, model_nb_grubbsPCA$results$ROC, model_nb_grubbsPCA$results$Sens, model_nb_grubbsPCA$results$Spec, model_nb_grubbsPCA$results$Accuracy, model_nb_grubbsPCA$results$Kappa, model_nb_grubbsPCA$results$AUC, model_nb_grubbsPCA$results$Precision, model_nb_grubbsPCA$results$Recall, model_nb_grubbsPCA$results$F))
  colnames(nb_results_rpca_grubbs) <- cbind('pAUC', 'MCC', 'ROC', 'Sens', 'Spec', 'Acc', 'Kappa', 'AUC', 'Precision', 'Recall', 'F') }
nb_results_rpca_grubbs <- nb_results_rpca_grubbs[which.max(nb_results_rpca_grubbs[,1]),]  # Keep the one with max pAUC amoung the different thresholds
nb_results_rpca_grubbs  # Show the result metrics

# The following part will apply PCA on the Rosner Outliers-removed Train Set, with different thresholds :
set.seed(7)
nb_results_rpca_rosner <- NULL  # Vector to store all metrics results
for(i in seq(70,95,5)){  # Looping over different thresholds for PCA
  control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                          preProcOptions = list(thresh = i/100), classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
  model_nb_rosnerPCA = train(y = outRosner_train$Label, x = x_reduced_train_rosner, method="nb",
                             preProcess=c("center", "scale", "pca"), trControl = control)
  nb_results_rpca_rosner <- rbind(nb_results_rpca_rosner, cbind(model_nb_rosnerPCA$results$pAUC, model_nb_rosnerPCA$results$MCC, model_nb_rosnerPCA$results$ROC, model_nb_rosnerPCA$results$Sens, model_nb_rosnerPCA$results$Spec, model_nb_rosnerPCA$results$Accuracy, model_nb_rosnerPCA$results$Kappa, model_nb_rosnerPCA$results$AUC, model_nb_rosnerPCA$results$Precision, model_nb_rosnerPCA$results$Recall, model_nb_rosnerPCA$results$F))
  colnames(nb_results_rpca_rosner) <- cbind('pAUC', 'MCC', 'ROC', 'Sens', 'Spec', 'Acc', 'Kappa', 'AUC', 'Precision', 'Recall', 'F') }
nb_results_rpca_rosner <- nb_results_rpca_rosner[which.max(nb_results_rpca_rosner[,1]),]  # Keep the one with max pAUC amoung the different thresholds
nb_results_rpca_rosner  # Show the result metrics


# -----  Summary of the different results :
res_nb <- rbind(nb_results_f, nb_results_grubbs, nb_results_rosner,
                nb_results_r, nb_results_rGrubbs, nb_results_rRosner,
                nb_results_rfe, nb_results_rfe_grubbs, nb_results_rfe_rosner)
rownames(res_nb) <- cbind('Full', 'Grubbs', 'Rosner', 'FullNoCorr', 'GrubbsNoCorr', 'rosnerNoCorr',
                          'FullRFE', 'GrubbsRFE', 'RosnerRFE')
res_nb <- res_nb[, c(11,10,1,2,3,4,5,6,7,8,9)]  # Reorder the columns to compare
resPca_nb <- rbind(nb_results_pca, nb_results_rpca, nb_results_rpca_grubbs, nb_results_rpca_rosner)
rownames(resPca_nb) <- cbind('FullPCA', 'ReducedPCA', 'GrubbsPCA', 'RosnerPCA')
# Show results :
scores_nb <- rbind(res_nb, resPca_nb)
scores_nb


# # ## ----  RESULTS - NB :  ----
#              pAUC   MCC    ROC   Sens  Spec  Accuracy Kappa AUC   Precision Recall F
# Full         0.789  0.590  0.823 0.848 0.729 0.788    0.577 NA    0.766     0.848  0.799
# Grubbs       0.777  0.563  0.829 0.856 0.698 0.769    0.543 NA    0.719     0.856  0.774
# Rosner       0.801  0.603  0.818 0.864 0.738 0.797    0.598 NA    0.763     0.864  0.804
# FullNoCorr   0.809  0.629  0.875 0.839 0.778 0.808    0.617 NA    0.799     0.839  0.812
# GrubbsNoCorr 0.818  0.646  0.870 0.814 0.821 0.817    0.632 NA    0.805     0.814  0.799
# rosnerNoCorr 0.804  0.611  0.870 0.859 0.750 0.801    0.606 0.160 0.773     0.859  0.807
# FullRFE      0.886  0.781  0.928 0.880 0.891 0.886    0.772 0.791 0.898     0.880  0.884
# GrubbsRFE    0.893  0.793  0.948 0.910 0.876 0.889    0.779 0.745 0.874     0.910  0.882
# RosnerRFE    0.886  0.783  0.959 0.863 0.908 0.887    0.773 0.778 0.905     0.863  0.877
# FullPCA      0.854  0.714  0.908 0.844 0.863 0.853    0.707 0.774 0.865     0.844  0.850
# ReducedPCA   0.866  0.737  0.913 0.867 0.864 0.866    0.731 0.776 0.867     0.867  0.864
# GrubbsPCA    0.829  0.670  0.885 0.806 0.853 0.828    0.655 0.680 0.837     0.806  0.808
# RosnerPCA    0.849  0.714  0.938 0.802 0.896 0.852    0.700 0.787 0.881     0.802  0.829

# Best NB : 
# GrubbsRFE    0.893  0.793  0.948 0.910 0.876 0.889    0.779 0.745 0.874     0.910  0.882

bestModel_NB <- model_nb_grubbsRFE


# ---------------------------------------------------------------------------------------------------


## -----------------------------------------------------------------------------
## --------------------------  RANDOM FOREST (RF)  -----------------------------
## -----------------------------------------------------------------------------

# ---  Define the RFE function for RF  ---
ctrl$functions <- rfFuncs
ctrl$functions$summary <- metricStats
set.seed(7)
# Run the RFE Algorithm, whose performance is based on AUC metric, for different sizes (number of possible features) :
rfRFE <- rfe(x=x_train_data, y=y_train, metric = "pAUC",
             sizes = c(4,8,10,12,14,16,18,20,22,24,50,100,200,300,400),
             preProc = c("center", "scale"), rfeControl = ctrl)
rfRFE  # Show the results
plot(rfRFE, type=c("g", "o"))  # Plot the AUC measures for different number of variables
rfPredict = predictors(rfRFE)  # Get the retained predictors
x_rf = subset(train_data, select = rfPredict)  # Keep only the RFE features for the Train Set

##  --->  We do the same steps for the Outliers removed Data (Both for Grubbs and Rosner) :
set.seed(7)  # Grubbs - Removed outliers
rfRFE_grubbs <- rfe(x=x_train_data_grubbs, y=factor(outGrubbs_train$Label),
                    sizes = c(4,8,10,12,14,16,18,20,22,24,50,100,200,300,400),
                    metric = "pAUC", preProc = c("center", "scale"), rfeControl = ctrl)
rfPredict_grubbs = predictors(rfRFE_grubbs)
x_rf_grubbs = subset(outGrubbs_train, select = rfPredict_grubbs)

set.seed(7)  # Rosner - Removed outliers
rfRFE_rosner <- rfe(x=x_train_data_rosner, y=factor(outRosner_train$Label),
                    sizes = c(4,8,10,12,14,16,18,20,22,24,50,100,200,300,400),
                    metric = "pAUC", preProc = c("center", "scale"), rfeControl = ctrl)
rfPredict_rosner = predictors(rfRFE_rosner)
x_rf_rosner = subset(outRosner_train, select = rfPredict_rosner)


## -----------------------------------------------------------------------------
##  ---------------   Comparing different RF Training   -----------------------

# -----------  RF on RFE Train Data :  -----------
# The following part will be applied on the whole Train Set :
set.seed(7)
rf_results_rfe <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_rf_fullRFE = train(y = y_train, x = x_rf, method="rf",
                         preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_rf_fullRFE$resample)-1))){  # Get the mean result
  rf_results_rfe <- cbind(rf_results_rfe, mean(model_rf_fullRFE$resample[, i]))
}
colnames(rf_results_rfe) <- names((model_rf_fullRFE$resample)[1:(length(model_rf_fullRFE$resample)-1)])
rf_results_rfe  # Show the result metrics


# The following part will be applied on the Grubbs' Outliers-removed Train Set :
set.seed(7)
rf_results_rfe_grubbs <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_rf_grubbsRFE = train(y = outGrubbs_train$Label, x = x_rf_grubbs, method="rf",
                           preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_rf_grubbsRFE$resample)-1))){
  rf_results_rfe_grubbs <- cbind(rf_results_rfe_grubbs, mean(model_rf_grubbsRFE$resample[, i]))
}
colnames(rf_results_rfe_grubbs) <- names((model_rf_grubbsRFE$resample)[1:(length(model_rf_grubbsRFE$resample)-1)])
rf_results_rfe_grubbs  # Show the result metrics


# The following part will be applied on the Rosner' Outliers-removed Train Set :
set.seed(7)
rf_results_rfe_rosner <- NULL
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_rf_rosnerRFE = train(y = outRosner_train$Label, x = x_rf_rosner, method="rf",
                           preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_rf_rosnerRFE$resample)-1))){
  rf_results_rfe_rosner <- cbind(rf_results_rfe_rosner, mean(model_rf_rosnerRFE$resample[, i]))
}
colnames(rf_results_rfe_rosner) <- names((model_rf_rosnerRFE$resample)[1:(length(model_rf_rosnerRFE$resample)-1)])
rf_results_rfe_rosner  # Show the result metrics



# -----------  RF on Full Train Data :  -----------
set.seed(7)
rf_results_f <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_rf_full = train(y = y_train, x = x_train_data, method="rf",
                      preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_rf_full$resample)-1))){
  rf_results_f <- cbind(rf_results_f, mean(model_rf_full$resample[, i]))
}
colnames(rf_results_f) <- names((model_rf_full$resample)[1:(length(model_rf_full$resample)-1)])
rf_results_f  # Show the result metrics

# -----------  RF on Outliers-removed Train Data :  -----------

# ------- Grubbs Outliers-removed Train set : -------
set.seed(7)
rf_results_grubbs <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_rf_grubbs = train(y = outGrubbs_train$Label, x = outGrubbs_train[, -length(train_data[,])], method="rf",
                        preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_rf_grubbs$resample)-1))){
  rf_results_grubbs <- cbind(rf_results_grubbs, mean(model_rf_grubbs$resample[, i]))
}
colnames(rf_results_grubbs) <- names((model_rf_grubbs$resample)[1:(length(model_rf_grubbs$resample)-1)])
rf_results_grubbs  # Show the result metrics

# ------- Rosner Outliers-removed Train set : -------
set.seed(7)
rf_results_rosner <- NULL  # Vector to store all metrics results
yt <- (outRosner_train['Label'])  # Label vector
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_rf_rosner = train(y = factor(yt[,]), x = outRosner_train[, -length(train_data[,])], method="rf",
                        preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_rf_rosner$resample)-1))){
  rf_results_rosner <- cbind(rf_results_rosner, mean(model_rf_rosner$resample[, i]))
}
colnames(rf_results_rosner) <- names((model_rf_rosner$resample)[1:(length(model_rf_rosner$resample)-1)])
rf_results_rosner  # Show the result metrics

# -----------  RF on Correlation-removed Train Data :  -----------
# Full Train Data (with outliers and no correlation)
set.seed(7)
rf_results_r <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_rf_noCorr = train(y = y_train, x = x_reduced_train, method="rf",
                        preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_rf_noCorr$resample)-1))){
  rf_results_r <- cbind(rf_results_r, mean(model_rf_noCorr$resample[, i]))
}
colnames(rf_results_r) <- names((model_rf_noCorr$resample)[1:(length(model_rf_noCorr$resample)-1)])
rf_results_r  # Show the result metrics

# Grubbs - Train Data (without outliers and no correlation)
set.seed(7)
rf_results_rGrubbs <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_rf_grubbsNoCorr = train(y = outGrubbs_train$Label, x = x_reduced_train_grubbs, method="rf",
                              preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_rf_grubbsNoCorr$resample)-1))){
  rf_results_rGrubbs <- cbind(rf_results_rGrubbs, mean(model_rf_grubbsNoCorr$resample[, i]))
}
colnames(rf_results_rGrubbs) <- names((model_rf_grubbsNoCorr$resample)[1:(length(model_rf_grubbsNoCorr$resample)-1)])
rf_results_rGrubbs  # Show the result metrics

# Rosner - Train Data (without outliers and no correlation)
set.seed(7)
rf_results_rRosner <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_rf_rosnerNoCorr = train(y = outRosner_train$Label, x = x_reduced_train_rosner, method="rf",
                              preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_rf_rosnerNoCorr$resample)-1))){
  rf_results_rRosner <- cbind(rf_results_rRosner, mean(model_rf_rosnerNoCorr$resample[, i]))
}
colnames(rf_results_rRosner) <- names((model_rf_rosnerNoCorr$resample)[1:(length(model_rf_rosnerNoCorr$resample)-1)])
rf_results_rRosner  # Show the result metrics


# -----------  RF on PCA Train Data :  -----------

# The following part will apply PCA on the whole Train Set, with different thresholds :
set.seed(7)
rf_results_pca <- NULL  # Vector to store all metrics results
for(i in seq(70,95,5)){  # Looping over different thresholds for PCA
  control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                          preProcOptions = list(thresh = i/100), classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
  model_rf_fullPCA = train(y = y_train, x = x_train_data, method="rf",
                           preProcess=c("center", "scale", "pca"), trControl = control)
  rf_results_pca <- rbind(rf_results_pca, cbind(model_rf_fullPCA$results$pAUC, model_rf_fullPCA$results$MCC, model_rf_fullPCA$results$ROC, model_rf_fullPCA$results$Sens, model_rf_fullPCA$results$Spec, model_rf_fullPCA$results$Accuracy, model_rf_fullPCA$results$Kappa, model_rf_fullPCA$results$AUC, model_rf_fullPCA$results$Precision, model_rf_fullPCA$results$Recall, model_rf_fullPCA$results$F))
  colnames(rf_results_pca) <- cbind('pAUC', 'MCC', 'ROC', 'Sens', 'Spec', 'Acc', 'Kappa', 'AUC', 'Precision', 'Recall', 'F') }
rf_results_pca <- rf_results_pca[which.max(rf_results_pca[,1]),]  # Keep the one with max pAUC amoung the different thresholds
rf_results_pca  # Show the result metrics


# The following part will apply PCA on the Correlation-removed Train Set, with different thresholds :
set.seed(7)
rf_results_rpca <- NULL  # Vector to store all metrics results
for(i in seq(70,95,5)){  # Looping over different thresholds for PCA
  control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                          preProcOptions = list(thresh = i/100), classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
  model_rf_rPCA = train(y = y_train, x = x_reduced_train, method="rf",
                        preProcess=c("center", "scale", "pca"), trControl = control)
  rf_results_rpca <- rbind(rf_results_rpca, cbind(model_rf_rPCA$results$pAUC, model_rf_rPCA$results$MCC, model_rf_rPCA$results$ROC, model_rf_rPCA$results$Sens, model_rf_rPCA$results$Spec, model_rf_rPCA$results$Accuracy, model_rf_rPCA$results$Kappa, model_rf_rPCA$results$AUC, model_rf_rPCA$results$Precision, model_rf_rPCA$results$Recall, model_rf_rPCA$results$F))
  colnames(rf_results_rpca) <- cbind('pAUC', 'MCC', 'ROC', 'Sens', 'Spec', 'Acc', 'Kappa', 'AUC', 'Precision', 'Recall', 'F') }
rf_results_rpca <- rf_results_rpca[which.max(rf_results_rpca[,1]),]  # Keep the one with max pAUC amoung the different thresholds
rf_results_rpca  # Show the result metrics

# The following part will apply PCA on the Grubbs Outliers-removed Train Set, with different thresholds :
set.seed(7)
rf_results_rpca_grubbs <- NULL  # Vector to store all metrics results
for(i in seq(70,95,5)){  # Looping over different thresholds for PCA
  control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                          preProcOptions = list(thresh = i/100), classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
  model_rf_grubbsPCA = train(y = outGrubbs_train$Label, x = x_reduced_train_grubbs, method="rf",
                             preProcess=c("center", "scale", "pca"), trControl = control)
  rf_results_rpca_grubbs <- rbind(rf_results_rpca_grubbs, cbind(model_rf_grubbsPCA$results$pAUC, model_rf_grubbsPCA$results$MCC, model_rf_grubbsPCA$results$ROC, model_rf_grubbsPCA$results$Sens, model_rf_grubbsPCA$results$Spec, model_rf_grubbsPCA$results$Accuracy, model_rf_grubbsPCA$results$Kappa, model_rf_grubbsPCA$results$AUC, model_rf_grubbsPCA$results$Precision, model_rf_grubbsPCA$results$Recall, model_rf_grubbsPCA$results$F))
  colnames(rf_results_rpca_grubbs) <- cbind('pAUC', 'MCC', 'ROC', 'Sens', 'Spec', 'Acc', 'Kappa', 'AUC', 'Precision', 'Recall', 'F') }
rf_results_rpca_grubbs <- rf_results_rpca_grubbs[which.max(rf_results_rpca_grubbs[,1]),]  # Keep the one with max pAUC amoung the different thresholds
rf_results_rpca_grubbs  # Show the result metrics

# The following part will apply PCA on the Rosner Outliers-removed Train Set, with different thresholds :
set.seed(7)
rf_results_rpca_rosner <- NULL  # Vector to store all metrics results
for(i in seq(70,95,5)){  # Looping over different thresholds for PCA
  control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                          preProcOptions = list(thresh = i/100), classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
  model_rf_rosnerPCA = train(y = outRosner_train$Label, x = x_reduced_train_rosner, method="rf",
                             preProcess=c("center", "scale", "pca"), trControl = control)
  rf_results_rpca_rosner <- rbind(rf_results_rpca_rosner, cbind(model_rf_rosnerPCA$results$pAUC, model_rf_rosnerPCA$results$MCC, model_rf_rosnerPCA$results$ROC, model_rf_rosnerPCA$results$Sens, model_rf_rosnerPCA$results$Spec, model_rf_rosnerPCA$results$Accuracy, model_rf_rosnerPCA$results$Kappa, model_rf_rosnerPCA$results$AUC, model_rf_rosnerPCA$results$Precision, model_rf_rosnerPCA$results$Recall, model_rf_rosnerPCA$results$F))
  colnames(rf_results_rpca_rosner) <- cbind('pAUC', 'MCC', 'ROC', 'Sens', 'Spec', 'Acc', 'Kappa', 'AUC', 'Precision', 'Recall', 'F') }
rf_results_rpca_rosner <- rf_results_rpca_rosner[which.max(rf_results_rpca_rosner[,1]),]  # Keep the one with max pAUC amoung the different thresholds
rf_results_rpca_rosner  # Show the result metrics


# -----  Summary of the different results :
res_rf <- rbind(rf_results_f, rf_results_grubbs, rf_results_rosner,
                rf_results_r, rf_results_rGrubbs, rf_results_rRosner,
                rf_results_rfe, rf_results_rfe_grubbs, rf_results_rfe_rosner)
rownames(res_rf) <- cbind('Full', 'Grubbs', 'Rosner', 'FullNoCorr', 'GrubbsNoCorr', 'rosnerNoCorr',
                          'FullRFE', 'GrubbsRFE', 'RosnerRFE')
res_rf <- res_rf[, c(11,10,1,2,3,4,5,6,7,8,9)]  # Reorder the columns to compare
resPca_rf <- rbind(rf_results_pca, rf_results_rpca, rf_results_rpca_grubbs, rf_results_rpca_rosner)
rownames(resPca_rf) <- cbind('FullPCA', 'ReducedPCA', 'GrubbsPCA', 'RosnerPCA')
# Show results :
scores_rf <- rbind(res_rf, resPca_rf)
scores_rf


# # ## ----  RESULTS - RF :  ----
#              pAUC   MCC    ROC   Sens  Spec  Accuracy Kappa AUC   Precision Recall F
# Full         0.869  0.747  0.924 0.851 0.887 0.870    0.739 0.785 0.890     0.851  0.865
# Grubbs       1.000  1.000  1.000 1.000 1.000 1.000    1.000 0.807 1.000     1.000  1.000
# Rosner       1.000  1.000  1.000 1.000 1.000 1.000    1.000 0.845 1.000     1.000  1.000
# FullNoCorr   0.878  0.765  0.928 0.863 0.892 0.878    0.756 0.782 0.895     0.863  0.874
# GrubbsNoCorr 0.858  0.729  0.905 0.850 0.867 0.857    0.712 0.686 0.859     0.850  0.840
# rosnerNoCorr 0.885  0.777  0.952 0.887 0.883 0.885    0.770 0.805 0.881     0.887  0.879
# FullRFE      0.896  0.800  0.939 0.900 0.891 0.896    0.792 0.806 0.900     0.900  0.896
# GrubbsRFE    0.886  0.781  0.937 0.886 0.886 0.884    0.768 0.728 0.883     0.886  0.873
# RosnerRFE    0.907  0.824  0.956 0.896 0.917 0.907    0.813 0.800 0.916     0.896  0.899
# FullPCA      0.851  0.712  0.930 0.827 0.875 0.851    0.701 0.793 0.876     0.827  0.844
# ReducedPCA   0.818  0.646  0.895 0.837 0.799 0.817    0.634 0.761 0.813     0.837  0.817
# GrubbsPCA    0.809  0.632  0.871 0.782 0.837 0.810    0.615 0.665 0.806     0.782  0.779
# RosnerPCA    0.827  0.668  0.926 0.812 0.842 0.828    0.653 0.781 0.839     0.812  0.808

# ---------------------------------------------------------------------------------------------------
# Best RF : 
# RosnerRFE    0.907  0.824  0.956 0.896 0.917 0.907    0.813 0.800 0.916     0.896  0.899

bestModel_RF <- model_rf_rosnerRFE


## -----------------------------------------------------------------------------
## --------------------  SUPPORT VECTOR MACHINES (SVM)  ------------------------
## -----------------------------------------------------------------------------

# ---  Define the RFE function for SVM  ---
cvCtrl <- trainControl(method = "cv", verboseIter = FALSE,
                       classProbs = TRUE, allowParallel = FALSE)
ctrl <- rfeControl(method = "repeatedcv", repeats = 5, number=10,
                   saveDetails = TRUE, returnResamp = "final")

ctrl$functions <- caretFuncs
ctrl$functions$summary <- metricStats

# Run the RFE Algorithm, whose performance is based on AUC metric, for different sizes (number of possible features) :
set.seed(7)
# Run the RFE Algorithm, whose performance is based on AUC metric, for different sizes (number of possible features) :
svmlRFE <- rfe(x=x_train_data, y=y_train, sizes = c(4,8,10,12,14,16,18,20,22,24,100,200,300,400),
               metric = "pAUC", rfeControl = ctrl,
               method = "svmLinear", preProc = c("center", "scale"), trControl = cvCtrl)

svmlRFE  # Show the results
trellis.device(color = 'blue')
plot(svmlRFE, type=c("g", "o"), main='ADvsCTL RFE', col='blue')  # Plot the AUC measures for different number of variables
svmlPredict = predictors(svmlRFE)  # Get the retained predictors
x_svml = subset(train_data, select = svmlPredict)  # Keep only the RFE features for the Train Set

##  --->  We do the same steps for the Outliers removed Data (Both for Grubbs and Rosner) :
set.seed(7)  # Grubbs - Removed outliers
svmlRFE_grubbs <- rfe(x=x_train_data_grubbs, y=factor(outGrubbs_train$Label), sizes = c(4,8,10,12,14,16,18,20,22,24,100,200,300,400),
                      metric = "pAUC", preProc = c("center", "scale"),
                      rfeControl = ctrl, trControl = cvCtrl)
svmlPredict_grubbs = predictors(svmlRFE_grubbs)
# plot(svmlRFE_grubbs, type=c("g", "o"))  # Plot the AUC measures for different number of variables
x_svml_grubbs = subset(outGrubbs_train, select = svmlPredict_grubbs)

set.seed(7)  # Rosner - Removed outliers
svmlRFE_rosner <- rfe(x=x_train_data_rosner, y=factor(outRosner_train$Label), sizes = c(4,8,10,12,14,16,18,20,22,24,100,200,300,400), 
                      metric = "pAUC", preProc = c("center", "scale"),
                      rfeControl = ctrl, trControl = cvCtrl)
svmlPredict_rosner = predictors(svmlRFE_rosner)
# plot(svmlRFE_rosner, type=c("g", "o"))  # Plot the AUC measures for different number of variables
x_svml_rosner = subset(outRosner_train, select = svmlPredict_rosner)


## -----------------------------------------------------------------------------
##  ---------------   Comparing different SVM Training   -----------------------

# -----------  SVM on RFE Train Data :  -----------
# The following part will be applied on the whole Train Set :
set.seed(7)
svml_results_rfe <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats)
model_svml_fullRFE = train(y = y_train, x = x_svml, method="svmLinear",
                           preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_svml_fullRFE$resample)-1))){  # Get the mean result
  svml_results_rfe <- cbind(svml_results_rfe, mean(model_svml_fullRFE$resample[, i]))
}
colnames(svml_results_rfe) <- names((model_svml_fullRFE$resample)[1:(length(model_svml_fullRFE$resample)-1)])
svml_results_rfe  # Show the result metrics

# The following part will be applied on the Grubbs' Outliers-removed Train Set :
set.seed(7)
svml_results_rfe_grubbs <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats)
model_svml_grubbsRFE = train(y = outGrubbs_train$Label, x = x_svml_grubbs, method="svmLinear",
                             preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_svml_grubbsRFE$resample)-1))){
  svml_results_rfe_grubbs <- cbind(svml_results_rfe_grubbs, mean(model_svml_grubbsRFE$resample[, i]))
}
colnames(svml_results_rfe_grubbs) <- names((model_svml_grubbsRFE$resample)[1:(length(model_svml_grubbsRFE$resample)-1)])
svml_results_rfe_grubbs  # Show the result metrics

# The following part will be applied on the Rosner' Outliers-removed Train Set :
set.seed(7)
svml_results_rfe_rosner <- NULL
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats)
model_svml_rosnerRFE = train(y = outRosner_train$Label, x = x_svml_rosner, method="svmLinear",
                             preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_svml_rosnerRFE$resample)-1))){
  svml_results_rfe_rosner <- cbind(svml_results_rfe_rosner, mean(model_svml_rosnerRFE$resample[, i]))
}
colnames(svml_results_rfe_rosner) <- names((model_svml_rosnerRFE$resample)[1:(length(model_svml_rosnerRFE$resample)-1)])
svml_results_rfe_rosner  # Show the result metrics


# -----------  svml on Full Train Data :  -----------
set.seed(7)
svml_results_f <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_svml_full = train(y = y_train, x = x_train_data, method="svmLinear",
                        preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_svml_full$resample)-1))){
  svml_results_f <- cbind(svml_results_f, mean(model_svml_full$resample[, i]))
}
colnames(svml_results_f) <- names((model_svml_full$resample)[1:(length(model_svml_full$resample)-1)])
svml_results_f  # Show the result metrics

# -----------  svm on Outliers-removed Train Data :  -----------

# ------- Grubbs Outliers-removed Train set : -------
set.seed(7)
svml_results_grubbs <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_svml_grubbs = train(y = outGrubbs_train$Label, x = outGrubbs_train[, -length(train_data[,])], method="svmLinear",
                          preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_svml_grubbs$resample)-1))){
  svml_results_grubbs <- cbind(svml_results_grubbs, mean(model_svml_grubbs$resample[, i]))
}
colnames(svml_results_grubbs) <- names((model_svml_grubbs$resample)[1:(length(model_svml_grubbs$resample)-1)])
svml_results_grubbs  # Show the result metrics

# ------- Rosner Outliers-removed Train set : -------
set.seed(7)
svml_results_rosner <- NULL  # Vector to store all metrics results
yt <- (outRosner_train['Label'])  # Label vector
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_svml_rosner = train(y = factor(yt[,]), x = outRosner_train[, -length(train_data[,])], method="svmLinear",
                          preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_svml_rosner$resample)-1))){
  svml_results_rosner <- cbind(svml_results_rosner, mean(model_svml_rosner$resample[, i]))
}
colnames(svml_results_rosner) <- names((model_svml_rosner$resample)[1:(length(model_svml_rosner$resample)-1)])
svml_results_rosner  # Show the result metrics

# -----------  svml on Correlation-removed Train Data :  -----------
# Full Train Data (with outliers and no correlation)
set.seed(7)
svml_results_r <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_svml_noCorr = train(y = y_train, x = x_reduced_train, method="svmLinear",
                          preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_svml_noCorr$resample)-1))){
  svml_results_r <- cbind(svml_results_r, mean(model_svml_noCorr$resample[, i]))
}
colnames(svml_results_r) <- names((model_svml_noCorr$resample)[1:(length(model_svml_noCorr$resample)-1)])
svml_results_r  # Show the result metrics

# Grubbs - Train Data (without outliers and no correlation)
set.seed(7)
svml_results_rGrubbs <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_svml_grubbsNoCorr = train(y = outGrubbs_train$Label, x = x_reduced_train_grubbs, method="svmLinear",
                                preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_svml_grubbsNoCorr$resample)-1))){
  svml_results_rGrubbs <- cbind(svml_results_rGrubbs, mean(model_svml_grubbsNoCorr$resample[, i]))
}
colnames(svml_results_rGrubbs) <- names((model_svml_grubbsNoCorr$resample)[1:(length(model_svml_grubbsNoCorr$resample)-1)])
svml_results_rGrubbs  # Show the result metrics

# Rosner - Train Data (without outliers and no correlation)
set.seed(7)
svml_results_rRosner <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_svml_rosnerNoCorr = train(y = outRosner_train$Label, x = x_reduced_train_rosner, method="svmLinear",
                                preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_svml_rosnerNoCorr$resample)-1))){
  svml_results_rRosner <- cbind(svml_results_rRosner, mean(model_svml_rosnerNoCorr$resample[, i]))
}
colnames(svml_results_rRosner) <- names((model_svml_rosnerNoCorr$resample)[1:(length(model_svml_rosnerNoCorr$resample)-1)])
svml_results_rRosner  # Show the result metrics


# -----------  svml on PCA Train Data :  -----------

# The following part will apply PCA on the whole Train Set, with different thresholds :
set.seed(7)
svml_results_pca <- NULL  # Vector to store all metrics results
for(i in seq(70,95,5)){  # Looping over different thresholds for PCA
  control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                          preProcOptions = list(thresh = i/100), classProbs = TRUE, summaryFunction = metricStats)
  model_svml_fullPCA = train(y = y_train, x = x_train_data, method="svmLinear",
                             preProcess=c("center", "scale", "pca"), trControl = control)
  svml_results_pca <- rbind(svml_results_pca, cbind(model_svml_fullPCA$results$pAUC, model_svml_fullPCA$results$MCC, model_svml_fullPCA$results$ROC, model_svml_fullPCA$results$Sens, model_svml_fullPCA$results$Spec, model_svml_fullPCA$results$Accuracy, model_svml_fullPCA$results$Kappa, model_svml_fullPCA$results$AUC, model_svml_fullPCA$results$Precision, model_svml_fullPCA$results$Recall, model_svml_fullPCA$results$F))
  colnames(svml_results_pca) <- cbind('pAUC', 'MCC', 'ROC', 'Sens', 'Spec', 'Acc', 'Kappa', 'AUC', 'Precision', 'Recall', 'F') }
svml_results_pca <- svml_results_pca[which.max(svml_results_pca[,1]),]  # Keep the one with max pAUC amoung the different thresholds
svml_results_pca  # Show the result metrics


# The following part will apply PCA on the Correlation-removed Train Set, with different thresholds :
set.seed(7)
svml_results_rpca <- NULL  # Vector to store all metrics results
for(i in seq(70,95,5)){  # Looping over different thresholds for PCA
  control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                          preProcOptions = list(thresh = i/100), classProbs = TRUE, summaryFunction = metricStats)
  model_svml_rPCA = train(y = y_train, x = x_reduced_train, method="svmLinear",
                          preProcess=c("center", "scale", "pca"), trControl = control)
  svml_results_rpca <- rbind(svml_results_rpca, cbind(model_svml_rPCA$results$pAUC, model_svml_rPCA$results$MCC, model_svml_rPCA$results$ROC, model_svml_rPCA$results$Sens, model_svml_rPCA$results$Spec, model_svml_rPCA$results$Accuracy, model_svml_rPCA$results$Kappa, model_svml_rPCA$results$AUC, model_svml_rPCA$results$Precision, model_svml_rPCA$results$Recall, model_svml_rPCA$results$F))
  colnames(svml_results_rpca) <- cbind('pAUC', 'MCC', 'ROC', 'Sens', 'Spec', 'Acc', 'Kappa', 'AUC', 'Precision', 'Recall', 'F') }
svml_results_rpca <- svml_results_rpca[which.max(svml_results_rpca[,1]),]  # Keep the one with max pAUC amoung the different thresholds
svml_results_rpca  # Show the result metrics

# The following part will apply PCA on the Grubbs Outliers-removed Train Set, with different thresholds :
set.seed(7)
svml_results_rpca_grubbs <- NULL  # Vector to store all metrics results
for(i in seq(70,95,5)){  # Looping over different thresholds for PCA
  control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                          preProcOptions = list(thresh = i/100), classProbs = TRUE, summaryFunction = metricStats)
  model_svml_grubbsPCA = train(y = outGrubbs_train$Label, x = x_reduced_train_grubbs, method="svmLinear",
                               preProcess=c("center", "scale", "pca"), trControl = control)
  svml_results_rpca_grubbs <- rbind(svml_results_rpca_grubbs, cbind(model_svml_grubbsPCA$results$pAUC, model_svml_grubbsPCA$results$MCC, model_svml_grubbsPCA$results$ROC, model_svml_grubbsPCA$results$Sens, model_svml_grubbsPCA$results$Spec, model_svml_grubbsPCA$results$Accuracy, model_svml_grubbsPCA$results$Kappa, model_svml_grubbsPCA$results$AUC, model_svml_grubbsPCA$results$Precision, model_svml_grubbsPCA$results$Recall, model_svml_grubbsPCA$results$F))
  colnames(svml_results_rpca_grubbs) <- cbind('pAUC', 'MCC', 'ROC', 'Sens', 'Spec', 'Acc', 'Kappa', 'AUC', 'Precision', 'Recall', 'F') }
svml_results_rpca_grubbs <- svml_results_rpca_grubbs[which.max(svml_results_rpca_grubbs[,1]),]  # Keep the one with max pAUC amoung the different thresholds
svml_results_rpca_grubbs  # Show the result metrics

# The following part will apply PCA on the Rosner Outliers-removed Train Set, with different thresholds :
set.seed(7)
svml_results_rpca_rosner <- NULL  # Vector to store all metrics results
for(i in seq(70,95,5)){  # Looping over different thresholds for PCA
  control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                          preProcOptions = list(thresh = i/100), classProbs = TRUE, summaryFunction = metricStats)
  model_svml_rosnerPCA = train(y = outRosner_train$Label, x = x_reduced_train_rosner, method="svmLinear",
                               preProcess=c("center", "scale", "pca"), trControl = control)
  svml_results_rpca_rosner <- rbind(svml_results_rpca_rosner, cbind(model_svml_rosnerPCA$results$pAUC, model_svml_rosnerPCA$results$MCC, model_svml_rosnerPCA$results$ROC, model_svml_rosnerPCA$results$Sens, model_svml_rosnerPCA$results$Spec, model_svml_rosnerPCA$results$Accuracy, model_svml_rosnerPCA$results$Kappa, model_svml_rosnerPCA$results$AUC, model_svml_rosnerPCA$results$Precision, model_svml_rosnerPCA$results$Recall, model_svml_rosnerPCA$results$F))
  colnames(svml_results_rpca_rosner) <- cbind('pAUC', 'MCC', 'ROC', 'Sens', 'Spec', 'Acc', 'Kappa', 'AUC', 'Precision', 'Recall', 'F') }
svml_results_rpca_rosner <- svml_results_rpca_rosner[which.max(svml_results_rpca_rosner[,1]),]  # Keep the one with max pAUC amoung the different thresholds
svml_results_rpca_rosner  # Show the result metrics


# -----  Summary of the different results :
res_svml <- rbind(svml_results_f, svml_results_grubbs, svml_results_rosner,
                  svml_results_r, svml_results_rGrubbs, svml_results_rRosner,
                  svml_results_rfe, svml_results_rfe_grubbs, svml_results_rfe_rosner)
rownames(res_svml) <- cbind('Full', 'Grubbs', 'Rosner', 'FullNoCorr', 'GrubbsNoCorr', 'rosnerNoCorr',
                            'FullRFE', 'GrubbsRFE', 'RosnerRFE')
res_svml <- res_svml[, c(11,10,1,2,3,4,5,6,7,8,9)]  # Reorder the columns to compare
resPca_svml <- rbind(svml_results_pca,svml_results_rpca,svml_results_rpca_grubbs, svml_results_rpca_rosner)
rownames(resPca_svml) <- cbind('FullPCA', 'ReducedPCA', 'GrubbsPCA', 'RosnerPCA')
# Show results :
scores_svml <- rbind(res_svml, resPca_svml)
scores_svml

# # ## ----  RESULTS - SVML :  ----
#              pAUC   MCC    ROC   Sens  Spec  Accuracy Kappa AUC   Precision Recall F
# Full         0.913  0.834  0.969 0.926 0.900 0.913    0.826 0.843 0.910     0.926  0.914
# Grubbs       0.814  0.637  0.929 0.792 0.835 0.814    0.625 0.726 0.806     0.792  0.789
# Rosner       0.863  0.736  0.962 0.860 0.867 0.863    0.726 0.819 0.863     0.860  0.855
# FullNoCorr   0.905  0.820  0.962 0.942 0.868 0.905    0.810 0.836 0.884     0.942  0.908
# GrubbsNoCorr 0.810  0.629  0.916 0.816 0.805 0.808    0.615 0.708 0.787     0.816  0.790
# rosnerNoCorr 0.840  0.687  0.952 0.846 0.833 0.839    0.678 0.807 0.827     0.846  0.830
# FullRFE      0.913  0.834  0.969 0.926 0.900 0.913    0.826 0.843 0.910     0.926  0.914
# GrubbsRFE    0.847  0.708  0.924 0.829 0.866 0.847    0.691 0.719 0.854     0.829  0.826
# RosnerRFE    0.879  0.773  0.961 0.846 0.912 0.881    0.760 0.818 0.907     0.846  0.867
# FullPCA      0.881  0.772  0.951 0.869 0.894 0.882    0.763 0.827 0.898     0.869  0.877
# ReducedPCA   0.874  0.757  0.932 0.905 0.843 0.874    0.748 0.804 0.859     0.905  0.877
# GrubbsPCA    0.821  0.655  0.907 0.798 0.844 0.825    0.644 0.708 0.822     0.798  0.801
# RosnerPCA    0.886  0.777  0.950 0.897 0.875 0.885    0.770 0.801 0.869     0.897  0.878

# Best SVM Linear :
# FullRFE      0.913  0.834  0.969 0.926 0.900 0.913    0.826 0.843 0.910     0.926  0.914

bestModel_SVML <- model_svml_fullRFE


## -----------------------------------------------------------------------------
## --------------------  SUPPORT VECTOR MACHINES (RBF)  ------------------------
## -----------------------------------------------------------------------------

# Here we follow the same steps for RBF
# ---  Define the RFE function for SVM  ---
cvCtrl <- trainControl(method = "cv", verboseIter = FALSE,
                       classProbs = TRUE, allowParallel = FALSE)
ctrl <- rfeControl(method = "repeatedcv", repeats = 5, number=10,
                   saveDetails = TRUE, returnResamp = "final")

ctrl$functions <- caretFuncs
ctrl$functions$summary <- metricStats

# Run the RFE Algorithm, whose performance is based on AUC metric, for different sizes (number of possible features) :
set.seed(7)
# Run the RFE Algorithm, whose performance is based on AUC metric, for different sizes (number of possible features) :
svmrRFE <- rfe(x=x_train_data, y=y_train, sizes = c(4,8,10,12,14,16,18,20,22,24,10,200,300,400),
               metric = "pAUC", rfeControl = ctrl,
               method = "svmRadial", preProc = c("center", "scale"), trControl = cvCtrl)

svmrRFE  # Show the results
plot(svmrRFE, type=c("g", "o"))  # Plot the AUC measures for different number of variables
svmrPredict = predictors(svmrRFE)  # Get the retained predictors
x_svmr = subset(train_data, select = svmrPredict)  # Keep only the RFE features for the Train Set

##  --->  We do the same steps for the Outliers removed Data (Both for Grubbs and Rosner) :
set.seed(7)  # Grubbs - Removed outliers
svmrRFE_grubbs <- rfe(x=x_train_data_grubbs, y=factor(outGrubbs_train$Label), sizes = c(4,8,10,12,14,16,18,20,22,24,100,200,300,400), 
                      metric = "pAUC", preProc = c("center", "scale"),
                      rfeControl = ctrl, trControl = cvCtrl)
svmrPredict_grubbs = predictors(svmrRFE_grubbs)
# plot(svmrRFE_grubbs, type=c("g", "o"))  # Plot the AUC measures for different number of variables
x_svmr_grubbs = subset(outGrubbs_train, select = svmrPredict_grubbs)

set.seed(7)  # Rosner - Removed outliers
svmrRFE_rosner <- rfe(x=x_train_data_rosner, y=factor(outRosner_train$Label), sizes = c(4,8,10,12,14,16,18,20,22,24,100,200,300,400),
                      metric = "pAUC", preProc = c("center", "scale"),
                      rfeControl = ctrl, trControl = cvCtrl)
svmrPredict_rosner = predictors(svmrRFE_rosner)
# plot(svmrRFE_rosner, type=c("g", "o"))  # Plot the AUC measures for different number of variables
x_svmr_rosner = subset(outRosner_train, select = svmrPredict_rosner)
plot(svmrRFE_rosner, type=c("g", "o"))

## -----------------------------------------------------------------------------
##  ---------------   Comparing different svmr Training   -----------------------

# -----------  svmr on RFE Train Data :  -----------
# The following part will be applied on the whole Train Set :
set.seed(7)
svmr_results_rfe <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats)
model_svmr_fullRFE = train(y = y_train, x = x_svmr, method="svmRadial",
                           preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_svmr_fullRFE$resample)-1))){  # Get the mean result
  svmr_results_rfe <- cbind(svmr_results_rfe, mean(model_svmr_fullRFE$resample[, i]))
}
colnames(svmr_results_rfe) <- names((model_svmr_fullRFE$resample)[1:(length(model_svmr_fullRFE$resample)-1)])
svmr_results_rfe  # Show the result metrics

# The following part will be applied on the Grubbs' Outliers-removed Train Set :
set.seed(7)
svmr_results_rfe_grubbs <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats)
model_svmr_grubbsRFE = train(y = outGrubbs_train$Label, x = x_svmr_grubbs, method="svmRadial",
                             preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_svmr_grubbsRFE$resample)-1))){
  svmr_results_rfe_grubbs <- cbind(svmr_results_rfe_grubbs, mean(model_svmr_grubbsRFE$resample[, i]))
}
colnames(svmr_results_rfe_grubbs) <- names((model_svmr_grubbsRFE$resample)[1:(length(model_svmr_grubbsRFE$resample)-1)])
svmr_results_rfe_grubbs  # Show the result metrics

# The following part will be applied on the Rosner' Outliers-removed Train Set :
set.seed(7)
svmr_results_rfe_rosner <- NULL
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats)
model_svmr_rosnerRFE = train(y = outRosner_train$Label, x = x_svmr_rosner, method="svmRadial",
                             preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_svmr_rosnerRFE$resample)-1))){
  svmr_results_rfe_rosner <- cbind(svmr_results_rfe_rosner, mean(model_svmr_rosnerRFE$resample[, i]))
}
colnames(svmr_results_rfe_rosner) <- names((model_svmr_rosnerRFE$resample)[1:(length(model_svmr_rosnerRFE$resample)-1)])
svmr_results_rfe_rosner  # Show the result metrics


# -----------  svmr on Full Train Data :  -----------
set.seed(7)
svmr_results_f <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_svmr_full = train(y = y_train, x = x_train_data, method="svmRadial",
                        preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_svmr_full$resample)-1))){
  svmr_results_f <- cbind(svmr_results_f, mean(model_svmr_full$resample[, i]))
}
colnames(svmr_results_f) <- names((model_svmr_full$resample)[1:(length(model_svmr_full$resample)-1)])
svmr_results_f  # Show the result metrics

# -----------  svmr on Outliers-removed Train Data :  -----------

# ------- Grubbs Outliers-removed Train set : -------
set.seed(7)
svmr_results_grubbs <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_svmr_grubbs = train(y = outGrubbs_train$Label, x = outGrubbs_train[, -length(train_data[,])], method="svmRadial",
                          preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_svmr_grubbs$resample)-1))){
  svmr_results_grubbs <- cbind(svmr_results_grubbs, mean(model_svmr_grubbs$resample[, i]))
}
colnames(svmr_results_grubbs) <- names((model_svmr_grubbs$resample)[1:(length(model_svmr_grubbs$resample)-1)])
svmr_results_grubbs  # Show the result metrics

# ------- Rosner Outliers-removed Train set : -------
set.seed(7)
svmr_results_rosner <- NULL  # Vector to store all metrics results
yt <- (outRosner_train['Label'])  # Label vector
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_svmr_rosner = train(y = factor(yt[,]), x = outRosner_train[, -length(train_data[,])], method="svmRadial",
                          preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_svmr_rosner$resample)-1))){
  svmr_results_rosner <- cbind(svmr_results_rosner, mean(model_svmr_rosner$resample[, i]))
}
colnames(svmr_results_rosner) <- names((model_svmr_rosner$resample)[1:(length(model_svmr_rosner$resample)-1)])
svmr_results_rosner  # Show the result metrics

# -----------  svmr on Correlation-removed Train Data :  -----------
# Full Train Data (with outliers and no correlation)
set.seed(7)
svmr_results_r <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_svmr_noCorr = train(y = y_train, x = x_reduced_train, method="svmRadial",
                          preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_svmr_noCorr$resample)-1))){
  svmr_results_r <- cbind(svmr_results_r, mean(model_svmr_noCorr$resample[, i]))
}
colnames(svmr_results_r) <- names((model_svmr_noCorr$resample)[1:(length(model_svmr_noCorr$resample)-1)])
svmr_results_r  # Show the result metrics

# Grubbs - Train Data (without outliers and no correlation)
set.seed(7)
svmr_results_rGrubbs <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_svmr_grubbsNoCorr = train(y = outGrubbs_train$Label, x = x_reduced_train_grubbs, method="svmRadial",
                                preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_svmr_grubbsNoCorr$resample)-1))){
  svmr_results_rGrubbs <- cbind(svmr_results_rGrubbs, mean(model_svmr_grubbsNoCorr$resample[, i]))
}
colnames(svmr_results_rGrubbs) <- names((model_svmr_grubbsNoCorr$resample)[1:(length(model_svmr_grubbsNoCorr$resample)-1)])
svmr_results_rGrubbs  # Show the result metrics

# Rosner - Train Data (without outliers and no correlation)
set.seed(7)
svmr_results_rRosner <- NULL  # Vector to store all metrics results
control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                        classProbs = TRUE, summaryFunction = metricStats, returnData = TRUE)
model_svmr_rosnerNoCorr = train(y = outRosner_train$Label, x = x_reduced_train_rosner, method="svmRadial",
                                preProcess=c("center", "scale"), trControl = control)
for(i in seq(1:(length(model_svmr_rosnerNoCorr$resample)-1))){
  svmr_results_rRosner <- cbind(svmr_results_rRosner, mean(model_svmr_rosnerNoCorr$resample[, i]))
}
colnames(svmr_results_rRosner) <- names((model_svmr_rosnerNoCorr$resample)[1:(length(model_svmr_rosnerNoCorr$resample)-1)])
svmr_results_rRosner  # Show the result metrics


# -----------  svmr on PCA Train Data :  -----------

# The following part will apply PCA on the whole Train Set, with different thresholds :
set.seed(7)
svmr_results_pca <- NULL  # Vector to store all metrics results
for(i in seq(70,95,5)){  # Looping over different thresholds for PCA
  control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                          preProcOptions = list(thresh = i/100), classProbs = TRUE, summaryFunction = metricStats)
  model_svmr_fullPCA = train(y = y_train, x = x_train_data, method="svmRadial",
                             preProcess=c("center", "scale", "pca"), trControl = control)
  svmr_results_pca <- rbind(svmr_results_pca, cbind(model_svmr_fullPCA$results$pAUC, model_svmr_fullPCA$results$MCC, model_svmr_fullPCA$results$ROC, model_svmr_fullPCA$results$Sens, model_svmr_fullPCA$results$Spec, model_svmr_fullPCA$results$Accuracy, model_svmr_fullPCA$results$Kappa, model_svmr_fullPCA$results$AUC, model_svmr_fullPCA$results$Precision, model_svmr_fullPCA$results$Recall, model_svmr_fullPCA$results$F))
  colnames(svmr_results_pca) <- cbind('pAUC', 'MCC', 'ROC', 'Sens', 'Spec', 'Acc', 'Kappa', 'AUC', 'Precision', 'Recall', 'F') }
svmr_results_pca <- svmr_results_pca[which.max(svmr_results_pca[,1]),]  # Keep the one with max pAUC amoung the different thresholds
svmr_results_pca  # Show the result metrics


# The following part will apply PCA on the Correlation-removed Train Set, with different thresholds :
set.seed(7)
svmr_results_rpca <- NULL  # Vector to store all metrics results
for(i in seq(70,95,5)){  # Looping over different thresholds for PCA
  control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                          preProcOptions = list(thresh = i/100), classProbs = TRUE, summaryFunction = metricStats)
  model_svmr_rPCA = train(y = y_train, x = x_reduced_train, method="svmRadial",
                          preProcess=c("center", "scale", "pca"), trControl = control)
  svmr_results_rpca <- rbind(svmr_results_rpca, cbind(model_svmr_rPCA$results$pAUC, model_svmr_rPCA$results$MCC, model_svmr_rPCA$results$ROC, model_svmr_rPCA$results$Sens, model_svmr_rPCA$results$Spec, model_svmr_rPCA$results$Accuracy, model_svmr_rPCA$results$Kappa, model_svmr_rPCA$results$AUC, model_svmr_rPCA$results$Precision, model_svmr_rPCA$results$Recall, model_svmr_rPCA$results$F))
  colnames(svmr_results_rpca) <- cbind('pAUC', 'MCC', 'ROC', 'Sens', 'Spec', 'Acc', 'Kappa', 'AUC', 'Precision', 'Recall', 'F') }
svmr_results_rpca <- svmr_results_rpca[which.max(svmr_results_rpca[,1]),]  # Keep the one with max pAUC amoung the different thresholds
svmr_results_rpca  # Show the result metrics

# The following part will apply PCA on the Grubbs Outliers-removed Train Set, with different thresholds :
set.seed(7)
svmr_results_rpca_grubbs <- NULL  # Vector to store all metrics results
for(i in seq(70,95,5)){  # Looping over different thresholds for PCA
  control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                          preProcOptions = list(thresh = i/100), classProbs = TRUE, summaryFunction = metricStats)
  model_svmr_grubbsPCA = train(y = outGrubbs_train$Label, x = x_reduced_train_grubbs, method="svmRadial",
                               preProcess=c("center", "scale", "pca"), trControl = control)
  svmr_results_rpca_grubbs <- rbind(svmr_results_rpca_grubbs, cbind(model_svmr_grubbsPCA$results$pAUC, model_svmr_grubbsPCA$results$MCC, model_svmr_grubbsPCA$results$ROC, model_svmr_grubbsPCA$results$Sens, model_svmr_grubbsPCA$results$Spec, model_svmr_grubbsPCA$results$Accuracy, model_svmr_grubbsPCA$results$Kappa, model_svmr_grubbsPCA$results$AUC, model_svmr_grubbsPCA$results$Precision, model_svmr_grubbsPCA$results$Recall, model_svmr_grubbsPCA$results$F))
  colnames(svmr_results_rpca_grubbs) <- cbind('pAUC', 'MCC', 'ROC', 'Sens', 'Spec', 'Acc', 'Kappa', 'AUC', 'Precision', 'Recall', 'F') }
svmr_results_rpca_grubbs <- svmr_results_rpca_grubbs[which.max(svmr_results_rpca_grubbs[,1]),]  # Keep the one with max pAUC amoung the different thresholds
svmr_results_rpca_grubbs  # Show the result metrics

# The following part will apply PCA on the Rosner Outliers-removed Train Set, with different thresholds :
set.seed(7)
svmr_results_rpca_rosner <- NULL  # Vector to store all metrics results
for(i in seq(70,95,5)){  # Looping over different thresholds for PCA
  control <- trainControl(method="repeatedcv", number=10, repeats=3, returnResamp = 'final',
                          preProcOptions = list(thresh = i/100), classProbs = TRUE, summaryFunction = metricStats)
  model_svmr_rosnerPCA = train(y = outRosner_train$Label, x = x_reduced_train_rosner, method="svmRadial",
                               preProcess=c("center", "scale", "pca"), trControl = control)
  svmr_results_rpca_rosner <- rbind(svmr_results_rpca_rosner, cbind(model_svmr_rosnerPCA$results$pAUC, model_svmr_rosnerPCA$results$MCC, model_svmr_rosnerPCA$results$ROC, model_svmr_rosnerPCA$results$Sens, model_svmr_rosnerPCA$results$Spec, model_svmr_rosnerPCA$results$Accuracy, model_svmr_rosnerPCA$results$Kappa, model_svmr_rosnerPCA$results$AUC, model_svmr_rosnerPCA$results$Precision, model_svmr_rosnerPCA$results$Recall, model_svmr_rosnerPCA$results$F))
  colnames(svmr_results_rpca_rosner) <- cbind('pAUC', 'MCC', 'ROC', 'Sens', 'Spec', 'Acc', 'Kappa', 'AUC', 'Precision', 'Recall', 'F') }
svmr_results_rpca_rosner <- svmr_results_rpca_rosner[which.max(svmr_results_rpca_rosner[,1]),]  # Keep the one with max pAUC amoung the different thresholds
svmr_results_rpca_rosner  # Show the result metrics


# -----  Summary of the different results :
res_svmr <- rbind(svmr_results_f, svmr_results_grubbs, svmr_results_rosner,
                  svmr_results_r, svmr_results_rGrubbs, svmr_results_rRosner,
                  svmr_results_rfe, svmr_results_rfe_grubbs, svmr_results_rfe_rosner)
rownames(res_svmr) <- cbind('Full', 'Grubbs', 'Rosner', 'FullNoCorr', 'GrubbsNoCorr', 'rosnerNoCorr',
                            'FullRFE', 'GrubbsRFE', 'RosnerRFE')
res_svmr <- res_svmr[, c(11,10,1,2,3,4,5,6,7,8,9)]  # Reorder the columns to compare
resPca_svmr <- rbind(svmr_results_pca, svmr_results_rpca, svmr_results_rpca_grubbs, svmr_results_rpca_rosner)
rownames(resPca_svmr) <- cbind('FullPCA', 'ReducedPCA', 'GrubbsPCA', 'RosnerPCA')
# Show results :
scores_svmr <- rbind(res_svmr, resPca_svmr)
scores_svmr


# # ## ----  RESULTS - SVM RBF :  ----
#              pAUC   MCC    ROC   Sens  Spec  Accuracy Kappa AUC   Precision Recall F
# Full         0.886  0.780  0.930 0.900 0.871 0.886    0.772 0.798 0.883     0.900  0.888
# Grubbs       0.827  0.666  0.909 0.809 0.846 0.827    0.651 0.697 0.820     0.809  0.801
# Rosner       0.879  0.769  0.953 0.878 0.879 0.879    0.757 0.808 0.880     0.878  0.871
# FullNoCorr   0.878  0.765  0.927 0.896 0.859 0.878    0.756 0.795 0.874     0.896  0.880
# GrubbsNoCorr 0.826  0.664  0.894 0.816 0.836 0.825    0.648 0.689 0.821     0.816  0.806
# rosnerNoCorr 0.874  0.760  0.948 0.878 0.871 0.874    0.748 0.801 0.871     0.878  0.866
# FullRFE      0.886  0.781  0.946 0.881 0.891 0.887    0.773 0.820 0.898     0.881  0.884
# GrubbsRFE    0.877  0.764  0.948 0.878 0.876 0.875    0.749 0.745 0.870     0.878  0.862
# RosnerRFE    0.892  0.798  0.958 0.860 0.925 0.894    0.786 0.814 0.920     0.860  0.882
# FullPCA      0.888  0.783  0.940 0.875 0.900 0.888    0.776 0.809 0.903     0.875  0.885
# ReducedPCA   0.867  0.743  0.937 0.839 0.894 0.868    0.734 0.811 0.893     0.839  0.859
# GrubbsPCA    0.858  0.731  0.941 0.922 0.794 0.852    0.707 0.746 0.812     0.922  0.854
# RosnerPCA    0.891  0.794  0.953 0.877 0.904 0.892    0.783 0.804 0.904     0.877  0.883
# 
# Best SVM RBF :
# RosnerRFE    0.892  0.798  0.958 0.860 0.925 0.894    0.786 0.814 0.920     0.860  0.882

bestModel_SVMR <- model_svmr_rosnerRFE


# ------------------------------------------------------------------------------
# Plot for Performance among the best models for Comparison :

fittedModels <- resamples(list(SVML = bestModel_SVML,
                               SVMR = bestModel_SVMR,
                               LDA = bestModel_LDA,
                               RF = bestModel_RF,
                               NB = bestModel_NB,
                               GLM = bestModel_GLM,
                               KNN = bestModel_KNN ))
fittedModels
trellis.par.set(caretTheme())
trellis.device()  # Plot
dotplot(fittedModels, metric = c('pAUC', 'MCC', 'Accuracy', 'ROC'))  # Dotplot for the specified metrics
bestScores <- svml_results_rfe  # The chosen best model's scores
bestScores
bestOverallModel <- bestModel_SVML  # The chosen best model


# ---  PREDICTIONS :  ---
length(svmlPredict)  # Show the number of predictors used for the best model
testIDs <- test_data[svmlPredict]  # Get the test data's IDs
pred <- predict(bestOverallModel, test_data[svmlPredict])  # Make predictions on the Test set
pred  # Show the resulting predictions

ADCTLres <- data.frame(cbind(test_data_$ID, as.character(pred)))  # Store the predictions
colnames(ADCTLres) <- cbind('ID', 'Pred')
ADCTLres


ADCTLfeat <- c()  # Get the best model's corresponsing Training predictors' indexes
for(i in 1:length(svmlPredict)){
  ADCTLfeat <- cbind(ADCTLfeat, which(names(train_data_)==svmlPredict[i]))
}
ADCTLfeat <- sort(as.vector(ADCTLfeat))  # Sort the Training predictors' indexes
ADCTLfeat


# ------------------------------------------------------------------- #
# -------  Save the Results in .RData format  -------
# save(ADCTLres, file = "ADCTLres.RData")
# save(ADCTLfeat, file = "ADCTLfeat.RData")
# ------------------------------------------------------------------- #



# ---  Resources :  ------------------------------------------------------------
# http://appliedpredictivemodeling.com/blog/tag/Simulation
# https://stats.stackexchange.com/questions/214387/results-from-rfe-function-caret-to-compute-average-metrics-r
# https://topepo.github.io/caret/recursive-feature-elimination.html
# https://github.com/cran/AppliedPredictiveModeling/blob/master/inst/chapters/19_Feature_Select.R
# https://www.guru99.com/r-random-forest-tutorial.html
# https://machinelearningmastery.com/feature-selection-with-the-caret-r-package/
# https://daviddalpiaz.github.io/r4sl/the-caret-package.html#classification
# https://rpubs.com/maulikpatel/229684
# https://statsandr.com/blog/outliers-detection-in-r/
# https://r-coder.com/boxplot-r/
# http://www.sthda.com/english/wiki/ggplot2-quick-correlation-matrix-heatmap-r-software-and-data-visualization
# https://bookdown.org/content/b298e479-b1ab-49fa-b83d-a57c2b034d49/correlation.html#heatmap
# https://machinelearningmastery.com/feature-selection-with-the-caret-r-package/

