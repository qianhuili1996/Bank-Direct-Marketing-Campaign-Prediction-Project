#title: "Reg_result_analysis"
#author: "Qianhui Li"

#-----------------------------------------------------

setwd("/Users/qianhuili/Desktop/GitHub/AAE724/Script/Data_cleaning")

library(tidyr)   
library(dplyr)   
library(leaps)
library(glmnet)
library(ggplot2)
library(gmodels)
library(MASS)
library(corrplot)
library(ISLR)
library(tree)
library(gridExtra)
library(ROCR)
library(rpart)
library(rpart.plot)
library(rattle)
library(pROC)
library(corrplot)
library(lfe)
library(car)
library(tidyverse)
library(viridis)

library(ggpubr)

library(plotly)
library(corrplot)
library(ROSE)
library(naniar)
library(caret)
library(blorr)
library(pROC)
#=============================================

##Data Preparation

bankoriginal<-read.csv("bank_data.csv",header=TRUE, sep=";", na.strings=c("unknown","non-existent"))

#----------
#Check # & % of missing values
gg_miss_var(bankoriginal)
gg_miss_var(bankoriginal, show_pct = TRUE)


#Since there is "999" in pdays means client was not previously contacted, I convert pdays into a dummy variable, never contacted(999)=0,others=1.

bankoriginal$pdays <-as.factor(bankoriginal$pdays)
bankoriginal$pdays <-ifelse(bankoriginal$pdays==999,0,1)


#The first variable that has the largest proportion of missing values is "default",
#However, it may be possible that customer is not willing to disclose this information to the banking representative. 
#Hence the unknown value in 'default' is actually a separate value.
#Thus I kept the variable "default", and I think it also make sense for "loan" and "housing" loan variable
bankoriginal$default <- as.character(bankoriginal$default)
bankoriginal$default[is.na(bankoriginal$default)] <- "refuse2disclose"

bankoriginal$loan<-as.character(bankoriginal$loan)
bankoriginal$loan[is.na(bankoriginal$loan)] <- "refuse2disclose"

bankoriginal$housing<-as.character(bankoriginal$housing)
bankoriginal$housing[is.na(bankoriginal$housing)] <- "refuse2disclose"
#As indicated by the data contributor, the duration is not known before a call is performed. 
#Also, after the end of the call y is obviously known. 
#Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
#Thus I removed "duration"
bankoriginal = bankoriginal %>% 
  select(-duration)

#check for missing value graph again
gg_miss_var(bankoriginal)
gg_miss_var(bankoriginal, show_pct = TRUE)

#omit missing values
bank<-na.omit(bankoriginal)
sum(is.na(bank))

#Data summary
summary(bank)


#convert variable types
sapply(bank,class)

#numerical variables
bank$age <- as.numeric(bank$age)
bank$campaign <- as.numeric(bank$campaign)
bank$previous <- as.numeric(bank$previous)
bank$emp.var.rate <- as.numeric(bank$emp.var.rate)
bank$cons.price.idx <- as.numeric(bank$cons.price.idx)
bank$cons.conf.idx <- as.numeric(bank$cons.conf.idx)
bank$euribor3m <- as.numeric(bank$euribor3m)
bank$nr.employed <- as.numeric(bank$nr.employed)

#categorical variables
bank$job <-as.factor(bank$job)
bank$marital <-as.factor(bank$marital)
bank$education <-as.factor(bank$education)
bank$default <-as.factor(bank$default)
bank$loan <-as.factor(bank$loan)
bank$housing<-as.factor(bank$housing)
bank$contact <-as.factor(bank$contact)
bank$poutcome <-as.factor(bank$poutcome)
bank$day_of_week <-as.factor(bank$day_of_week)
bank$month <-as.factor(bank$month)
bank$pdays <-as.factor(bank$pdays)

bank$y<-ifelse(bank$y =='yes',1,0)
bank$y <-as.factor(bank$y)

#---------------------
#Remove outliers
#age==38770
x <- bank$age
qnt <- quantile(x, probs=c(.25, .75), na.rm = T)
H <- 1.5 * IQR(x, na.rm = T)
hb <- H + qnt[2]
hb #remove>69.5
ab <- bank[which(bank$age<hb),]

#campaign==35982
x1 <- bank$campaign
qnt1 <- quantile(x1, probs=c(.25, .75), na.rm = T)
H1 <- 1.5 * IQR(x1, na.rm = T)
hb1<- H1 + qnt1[2]
hb1 #remove>6
ac <- bank[which(bank$campaign<hb1),]

#cons.conf.idx
x5 <- bank$cons.conf.idx 
qnt5 <- quantile(x5, probs=c(.25, .75), na.rm = T)
H5 <- 1.5 * IQR(x5, na.rm = T)
hb5<- H5 + qnt5[2]
hb5 #remove>-26.95

#From the boxplot for "previous", I decided to treat observations larger than 2 as outliers, thus remove them.

#Result after removing outliers in numerical variables(34,370obs with 20 variables)
bank <- bank[which(bank$age<hb & bank$campaign<hb1 & bank$previous<2 & bank$cons.conf.idx<hb5),]

table(bank$education)
bank=bank[!as.factor(bank$education) %in% which(table(bank$education)=="illiterate"),]
bank$education=droplevels(bank$education)
#From the histogram, there is one obvious tiny number of counts for "illterate"(16 observations)
#Thus I decided to drop obs with "illiterate"

#After removing outliers for both numerical and categorical variables, there are 34,354 obs with 20 variables.

#Check data imbalance
counts <- table(bank$y)
barplot(counts,col=c("royalblue3","tomato3"),legend = rownames(counts), main = "Original Customers' Responses")
CrossTable(bank$y)

#SMOTE
library(DMwR)
set.seed(29650634)
balanced <- SMOTE(y ~ ., bank, perc.over = 400, perc.under=100)


counts1 <- table(balanced$y)
barplot(counts1,col=c("royalblue3","tomato3"),legend = rownames(counts), main = "Balanced Customers' Responses")
CrossTable(balanced$y)
#=============================================

#Data Split for Logistic Regression, CART, and Random Forest

set.seed(84798034)
index <- createDataPartition(balanced$y, p = 0.5, list = FALSE)
train_data <- balanced[index, ]
test_data  <- balanced[-index, ] 



#===========================================================
#Predictions

#Logistic Regression

## Feature selection: For logistic regression and neural nets, use 13 variables
modelk <- glm(y ~ ., data = balanced, family = binomial(link = 'logit'))

modelk %>% blr_step_aic_both() %>%plot()

subsettrain<-c("y","nr.employed","poutcome","loan", "month", "pdays","default", "previous", 
"marital", "housing", "cons.conf.idx", "education", "job", "day_of_week",
"campaign","age", "euribor3m","emp.var.rate","cons.price.idx")
train_lognn <-train_data[subsettrain]

subsettest<-c("y","nr.employed","poutcome","loan", "month", "pdays","default", "previous", "marital", "housing", "cons.conf.idx", "education", "job", "day_of_week",
              "campaign","age", "euribor3m","emp.var.rate","cons.price.idx")
test_lognn<-test_data[subsettest]


set.seed(859837)
logit_model <- glm(y ~.,family=binomial(link='logit'),data =train_lognn)
summary(logit_model)
anova(logit_model, test="Chisq")

#variable importance
km<-varImp(logit_model)

library(data.table)
setDT(km, keep.rownames = TRUE)[]
colnames(km)[2]<-"Importance"
colnames(km)[1]<-"Variables"
km %>%
  arrange(Importance) %>%    # First sort by val. This sort the dataframe but NOT the factor levels
  mutate(name=fct_reorder(Variables, desc(Importance))) %>%   # This trick update the factor levels
  ggplot( aes(x=Variables, y=Importance)) +
  geom_segment( aes(xend=name, yend=0)) +
  geom_point( size=4, color="orange") +
  coord_flip() +
  theme_bw() +
  xlab("")


#confusion matrix for train
log.pred.train <-predict(logit_model,data=train_lognn,type="response")
log.pred1.train <-ifelse(log.pred.train>0.5,1,0)
log.confusion.matrix.train <-table(log.pred1.train,train_lognn$y)
log.confusion.matrix.train

log.accuracy.train=sum(diag(log.confusion.matrix.train))/sum(log.confusion.matrix.train)
log.accuracy.train
#0.7596703


#confusion matrix for test
log.pred.test <-predict(logit_model,data=test_lognn,type="response")
log.pred1.test <-ifelse(log.pred.test>0.5,1,0)
Log_cm <- table(log.pred1.test, test_lognn$y)
Log_cm

log.accuracy.test=sum(diag(Log_cm))/sum(Log_cm)
log.accuracy.test
#0.7596703

#ROC-AUC logistic
  #Train
library(ROCR)

detach(package:neuralnet)
pred <- prediction(log.pred.train, train_lognn$y)  
perf <- performance(pred,"tpr","fpr")
plot(perf, main = "Logistic ROC_train", colorize=TRUE)
abline(0,1)

AUC_log_tr <- auc(roc(train_lognn$y, log.pred.train))
AUC_log_tr 
#0.8298

  #Test
predl <- prediction(log.pred.test, test_lognn$y)  
perfl <- performance(predl,"tpr","fpr")
plot(perfl, main = "Logistic ROC_test", colorize=TRUE)
abline(0,1)

AUC_log_te <- auc(roc(test_lognn$y, log.pred.test))
AUC_log_te
#0.8298
#=================

###CART
set.seed(34857802)
tree_model <- rpart(y ~ ., data = train_data,method="class")
tree_model
fancyRpartPlot(tree_model)

#predict train
predictions <- predict(tree_model, train_data, type = "class")

#confusion matrix train
tree.confusion.matrix.train <- prop.table(table(predictions, train_data$y))
tree.confusion.matrix.train

CrossTable(train_data$y, predictions,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual subscription status', 'predicted subscription status'))

#train accuracy
tree.accuracy.train=sum(diag(tree.confusion.matrix.train))/sum(tree.confusion.matrix.train)
tree.accuracy.train


#predict test
cart_pred <- predict(tree_model , test_data,type="class")



# Confusion matrix for test
tree.confusion.matrix.test <- prop.table(table(cart_pred, test_data$y))
tree.confusion.matrix.test

#test accuracy
tree.accuracy.test=sum(diag(tree.confusion.matrix.test))/sum(tree.confusion.matrix.test)
tree.accuracy.test

# Cross table validation for test
CrossTable(test_data$y, cart_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual customers responses', 'predicted customers responses'))

##prune tree
set.seed(579837)

printcp(tree_model)
plotcp(tree_model)

tree_model$cptable[which.min(tree_model$cptable[,"xerror"]),"CP"]

bestcp <- tree_model$cptable[which.min(tree_model$cptable[,"xerror"]),"CP"]
tree.pruned <- prune(tree_model, cp = bestcp)

fancyRpartPlot(tree.pruned)
summary(tree.pruned)

#variable importance
argPlot <- as.data.frame(tree.pruned$variable.importance)
df <- data.frame(imp = tree.pruned$variable.importance)
df2 <- df %>% 
  tibble::rownames_to_column() %>% 
  dplyr::rename("variable" = rowname) %>% 
  dplyr::arrange(imp) %>%
  dplyr::mutate(variable = forcats::fct_inorder(variable))

ggplot2::ggplot(df2) +
  geom_segment(aes(x = variable, y = 0, xend = variable, yend = imp), 
               size = 1.5, alpha = 0.7) +
  geom_point(aes(x = variable, y = imp, col = variable), 
             size = 4, show.legend = F)  +
  coord_flip() + labs(x="Variables",y="Importance")+
  theme_bw()

# Compute the train accuracy of the pruned tree

predtree_train<- predict(tree.pruned, train_data, type = "class")
accuracy_prun_train <- mean(predtree_train == train_data$y)
accuracy_prun_train
pruned.confusion.matrix.train <- prop.table(table(predtree_train, train_data$y))
pruned.confusion.matrix.train
#0.8919393

# Compute the test accuracy of the pruned tree
predtree_test <- predict(tree.pruned, test_data, type = "class")
accuracy_prune_test <- mean(predtree_test== test_data$y)
accuracy_prune_test
pruned.confusion.matrix.test <- prop.table(table(predtree_test, test_data$y))
pruned.confusion.matrix.test
#0.8915498

#The tree after being pruned is the same as before

#ROC-AUC tree
#Train
library(ROCR)

pred2 <- prediction(as.numeric(predtree_train), as.numeric(train_data$y) ) 
perf2 <- performance(pred2,"tpr","fpr")
plot(perf2, main = "Tree ROC_train", colorize=TRUE)
abline(0,1)

AUC_tree_tr <- auc(roc(as.numeric(train_data$y), as.numeric(predtree_train)))
AUC_tree_tr 
#0.8963

#Test
pred3 <- prediction(as.numeric(predtree_test), test_data$y)  
perf3 <- performance(pred3,"tpr","fpr")
plot(perf3, main = "Tree ROC_test", colorize=TRUE)
abline(0,1)

AUC_log_te <- auc(roc(as.numeric(test_data$y), as.numeric(predtree_test)))
AUC_log_te
#0.895
#=================


#=============================================================

#Random Forest
library(randomForest)
set.seed(3078684)
RF.model <- randomForest(y~., data=train_data, ntree=100, importance=TRUE)
RF.model
summary(RF.model)

#variable importance
varImpPlot(RF.model)

#partial dependece plot

partialPlot(RF.model,train_data,cons.conf.idx,which.class = 1)
partialPlot(RF.model,train_data,cons.price.idx,which.class = 1)
partialPlot(RF.model,train_data,nr.employed,which.class = 1)


#library(randomForestExplainer)
#explain_forest(RF.model, interactions = F, data = train_data)

k <- 10
getTree(RF.model, k, labelVar = TRUE)

#Next we display an error plot of the random forest model:
plot(RF.model)

#train
RF.predict.train <- predict(RF.model, newdata = train_data)
RF.train.cm <- as.matrix(table(Actual1 = train_data$y, Predicted1 = RF.predict.train))
RF.train.cm
accuracy_train_rf=sum(diag(RF.train.cm))/sum(RF.train.cm)
accuracy_train_rf
#0.9609294

#test
RF.predict <- predict(RF.model, newdata = test_data)
RF.cm <- as.matrix(table(Actual = test_data$y, Predicted = RF.predict))
RF.cm

accuracy_test_rf=sum(diag(RF.cm))/sum(RF.cm)
accuracy_test_rf
#0.9160826

#Below we test the accuracy on the training and test datasets and we see that it is 90.87% and 83.45%, respectively. 
#The “out of sample” error is 16.51% and is in agreement with the OOB error:

#library(randomForestExplainer)
#explain_forest(RF.model, interactions = TRUE, data = test_data)


#RF ROC-AUC
  #train
RFpred <- prediction(as.numeric(RF.predict.train), as.numeric(train_data$y))
RFperf <- performance(RFpred, 'tpr','fpr')

plot(RFperf, main = "RF ROC_train", colorize=TRUE)
abline(0,1)
AUC_RF_tr <- auc(roc(as.numeric(train_data$y), as.numeric(RF.predict.train)))
AUC_RF_tr 
#0.965
  #test
RFpred1 <- prediction(as.numeric(RF.predict), as.numeric(test_data$y))
RFperf1 <- performance(RFpred1, 'tpr','fpr')
plot(RFperf1, main = "RF ROC_test", colorize=TRUE)
abline(0,1)
AUC_RF_te <- auc(roc(as.numeric(test_data$y), as.numeric(RF.predict)))
AUC_RF_te 
#0.9157


#==============================================================

#neural nets
library(nnet)
library(NeuralNetTools)
library(neuralnet)

## For neural nets, use -- variables(Feature selection step)

#I don't inclue "y" variable, 
#because I dont want to make it split into 2 column y0 and y1 in 
#in one-hot-encoding
subsetnn <- subset(balanced, select=c("y","nr.employed","poutcome","loan", "month", "pdays","default", "previous", 
                                       "marital", "housing", "cons.conf.idx", "education", "job", "day_of_week",
                                       "campaign","age", "euribor3m","emp.var.rate","cons.price.idx"))

subset_y <- balanced$y

#Use of one-hot-encoding to transfer categorical variables into numerical variables
#Since I want to do data scaling before applying nn, 
#and it requires numerical variables
dmy <- dummyVars(" ~ .", data = subsetnn)
bank.dummies<- data.frame(predict(dmy, newdata = subsetnn))

## Scale data for neural network
max = apply(bank.dummies , 2 , max)
min = apply(bank.dummies, 2 , min)
scaled = as.data.frame(scale(bank.dummies, center = min, scale = max - min))

#add back the y variable back to bank.dummies
scaled$y <- as.factor(subset_y)

##data_split

idx <- sample(1:dim(scaled)[1], dim(scaled)[1]/2)
trainNN = scaled[idx , ]
trainNN = trainNN %>% 
  select(-y.0,-y.1)
testNN = scaled[-idx , ]
testNN = testNN %>% 
  select(-y.0,-y.1)
#model fit
set.seed(2304862)
ctrl<-trainControl(method="repeatedcv",repeats=5,classProbs=TRUE)
nn <- train(y ~ .,
            data = trainNN,
            method = "nnet",metric="Accuracy",maxit=1000)
print(nn)
plotnet(nn)

#variable importance(garson belongs to NeuralNetTools)

garson(nn$finalModel)+theme(axis.text.x     = element_text(angle = 45, hjust = 1, size=10))

#train
nnpredtrain <- predict(nn, trainNN)
resulttrainnn <-table(predicted=nnpredtrain,true=trainNN$y)
resulttrainnn
acctrainnn =sum(diag(resulttrainnn))/sum(resulttrainnn)
acctrainnn
#0.9033619

#test
nnpredtest <- predict(nn, testNN)
resulttestnn <-table(predicted=nnpredtest,true=testNN$y)
resulttestnn
acctestnn =sum(diag(resulttestnn))/sum(resulttestnn)
acctestnn
#0.8916796

#NN ROC-AUC
#train
detach(package:neuralnet)
NNpred <- prediction(as.numeric(nnpredtrain), as.numeric(trainNN$y))
NNperf <- performance(NNpred, 'tpr','fpr')

plot(NNperf, main = "NN ROC_train", colorize=TRUE)
abline(0,1)
AUC_NN_tr <- auc(roc(as.numeric(trainNN$y), as.numeric(nnpredtrain)))
AUC_NN_tr 
#0.8991

#test
NNpred1 <- prediction(as.numeric(nnpredtest), as.numeric(testNN$y))
NNperf1 <- performance(NNpred1, 'tpr','fpr')
plot(NNperf1, main = "NN ROC_test", colorize=TRUE)
abline(0,1)
AUC_NN_te <- auc(roc(as.numeric(testNN$y), as.numeric(nnpredtest)))
AUC_NN_te 
#0.8902


png("nn.png",height=2500, width=3000) 
plot(nn) 
dev.off()






#=============================================================


