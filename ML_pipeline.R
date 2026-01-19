# Machine Learning Pipeline for DKD Classification
############################################################

set.seed(123)

library(dplyr)
library(ggplot2)
library(pROC)
library(caret)
library(glmnet)
library(randomForest)
library(e1071)
library(xgboost)
library(keras3)

############################################################
# 1pre. data wash
############################################################
raw <- read.table("Dataset.txt", sep="\t", header=FALSE,check.names=FALSE)
#set colnames and rownames in expr: makesure colnames are sampleID and rownames are gene name
colnames(raw)<-raw[1,]
raw<-raw[-1, ]
raw1<-raw[-1,]

Dataset_clean<-raw1[!duplicated(raw1[,1]), ]
#if we want to use gene name as row names, gene names should be unique
Dataset_clean<-raw1[!duplicated(raw1[,1]), ]
rownames(Dataset_clean)<-Dataset_clean[,1]
Dataset_clean<-Dataset_clean[,-1]
View(Dataset_clean)



write.table(Dataset_clean, file="dataset_clean.txt", sep="\t",quote=FALSE)

############################################################
# 1. Load data
############################################################

sample <- read.table("sample.txt", sep="\t", header=FALSE)
colnames(sample) <- c("SampleID","Group")

hub <- read.table("hup_gene.txt")[,1]


expr <- read.table("Dataset_clean.txt", sep="\t", header=TRUE, row.names=1, check.names=FALSE)

# there are some sampleIDs only shown in expr not in sample.txt, remove these samples and only keep the overlap samples(common_samples) in expr and sample.txt
#find common samples

common_samples<-intersect(sample$SampleID, colnames(expr))
View(common_samples)
#clean sampleID only in sample.txt but not in expr
sample_clean<-sample[sample$SampleID %in% common_samples,]

#clean sampleID only in expr but not in sample.txt
expr_clean<-expr[,common_samples]

#check if it is consistant now (sampleID in expr and sampleID in sample.txt)
all(sample_clean$SampleID == colnames(expr_clean))
# adjust sampleID order in expr consistent with sampleID order in sample.txt
expr <- expr_clean[, sample_clean$SampleID]


############################################################
# 2. Prepare training & test set
############################################################

library(caret)
#tranpose sampleID from column name to row name (this is the typical format for machine learning)
X <- t(expr_clean)
#convert the group labels (e.g., DKD/control) from character strings to a factor. This tells machine learning models that the outcome is a categorical classfication task,
#rather than plain text or a continuous variable.
# Packages like caret and glmnet require the outcome variable(y) to be a factor for classfication

y <- factor(sample_clean$Group)

#'CreateDataPartition' ensures that the class distribution (DKD vs Control is kept the same as the DKD and control sample ratio in the sample dataset)
#p=0.8 means 80% of the samples are selected for training, list = FALSE forces the function to return a plain integer vector of row indices, instead of a list.
#This makes it possible to directly index data (e.g., X[train_idx, ]). 

train_idx <- createDataPartition(y, p=0.8, list=FALSE)
train_X <- X[train_idx, ]
test_X  <- X[-train_idx, ]

train_y <- y[train_idx]
test_y  <- y[-train_idx]

############################################################
# 3. Utility metrics
############################################################

get_metrics <- function(prob, true_label){
  pred <- ifelse(prob > 0.5, levels(true_label)[2], levels(true_label)[1])
  pred <- factor(pred, levels=levels(true_label))
  cm <- confusionMatrix(pred, true_label)
  list(
    ACC  = cm$overall["Accuracy"],
    F1   = cm$byClass["F1"],
    Sens = cm$byClass["Sensitivity"],
    Spec = cm$byClass["Specificity"],
    AUC  = auc(roc(as.numeric(true_label=="DKD"), prob))
  )
}

############################################################
# 4. LASSO
############################################################
 
library(glmnet)

cvfit <- cv.glmnet(as.matrix(train_X), train_y, family="binomial", alpha=1)
lasso_genes <- rownames(coef(cvfit, s="lambda.min"))[coef(cvfit, s="lambda.min")[,1]!=0]
lasso_genes <- setdiff(lasso_genes, "(Intercept)")
train_X_sel <- train_X[, lasso_genes, drop=FALSE]
test_X_sel  <- test_X[,  lasso_genes, drop=FALSE]

############################################################
# 5. Logistic Regression
############################################################
library(pROC)
glm_model <- glm(train_y ~ ., data=data.frame(train_y=train_y, train_X_sel), family=binomial)
glm_prob <- predict(glm_model, newdata=data.frame(test_X_sel), type="response")
glm_res <- get_metrics(glm_prob, test_y)

############################################################
# 6. Random Forest
############################################################

library(randomForest)
rf_model <- randomForest(train_X_sel, train_y, ntree=300)
rf_prob <- predict(rf_model, newdata=test_X_sel, type="prob")[,2]
rf_res <- get_metrics(rf_prob, test_y)

############################################################
# 7. SVM
############################################################

library(e1071)
svm_model <- svm(train_X_sel, train_y, kernel="radial", probability=TRUE)
svm_prob <- attr(predict(svm_model, newdata=test_X_sel, probability=TRUE), "probabilities")[,2]
library(caret)

library(pROC)
svm_res <- get_metrics(svm_prob, test_y)

############################################################
# 8. XGBoost
############################################################

library(xgboost)
dtrain <- xgb.DMatrix(as.matrix(train_X_sel), label=as.numeric(train_y=="DKD"))
dtest  <- xgb.DMatrix(as.matrix(test_X_sel))

xgb_model <- xgb.train(
  params=list(objective="binary:logistic", eta=0.1, max_depth=4, eval_metric="auc"),
  data=dtrain,
  nrounds=150
)

xgb_prob <- predict(xgb_model, newdata=dtest)
xgb_res <- get_metrics(xgb_prob, test_y)

############################################################
# 9. MLP Neural Network
############################################################

library(nnet)

# 1 hidden layer MLP
mlp_model <- nnet(
  x = as.matrix(train_X_sel),
  y = as.numeric(train_y == "DKD"),
  size = 16,      # hidden units
  maxit = 200,
  linout = FALSE, # logistic output
  decay = 5e-4
)

# prediction
library(caret)
library(pROC)
mlp_prob <- predict(mlp_model, as.matrix(test_X_sel), type = "raw")
mlp_prob <- as.numeric(mlp_prob)

mlp_res <- get_metrics(mlp_prob, test_y)
mlp_res

############################################################
# 10. Print Results
############################################################

cat("\n\n===== LASSO Selected Genes =====\n")
print(lasso_genes)

cat("\n\n===== Model Performance =====\n")
print(list(
  Logistic = glm_res,
  RandomForest = rf_res,
  SVM = svm_res,
  XGBoost = xgb_res,
  MLP = mlp_res
))
#results output
results_df <- data.frame(
  Model = c("Logistic", "RandomForest", "SVM", "XGBoost", "MLP"),
  ACC = c(glm_res$ACC, rf_res$ACC, svm_res$ACC, xgb_res$ACC, mlp_res$ACC),
  F1  = c(glm_res$F1,  rf_res$F1,  svm_res$F1,  xgb_res$F1,  mlp_res$F1),
  Sens = c(glm_res$Sens, rf_res$Sens, svm_res$Sens, xgb_res$Sens, mlp_res$Sens),
  Spec = c(glm_res$Spec, rf_res$Spec, svm_res$Spec, xgb_res$Spec, mlp_res$Spec),
  AUC  = c(
    as.numeric(glm_res$AUC),
    as.numeric(rf_res$AUC),
    as.numeric(svm_res$AUC),
    as.numeric(xgb_res$AUC),
    as.numeric(mlp_res$AUC)
  )
)
write.table(
  results_df,
  file = "model_performance.txt",
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

############################################################
# 11. ROC Plot
############################################################
library(pROC)
dir.create("results", showWarnings = FALSE)

png(
  filename = "roc_curves.png",
  width = 1800,    
  height = 1800,   
  res = 300        
)


plot(
  roc(test_y, glm_prob),
  col = "black",
  main = "ROC Curves"
)

lines(roc(test_y, rf_prob), col = "blue")
lines(roc(test_y, svm_prob), col = "red")
lines(roc(test_y, xgb_prob), col = "purple")
lines(roc(test_y, mlp_prob), col = "green")

legend(
  "bottomright",
  legend = c("GLM", "RF", "SVM", "XGB", "MLP"),
  col = c("black", "blue", "red", "purple", "green"),
  lwd = 2
)


dev.off()
