f<-"C:\\Users\\patel\\OneDrive\\Desktop\\Data_MultiSlice_CompanyX.xlsx"
library(readxl) #package to read from excel
library(dplyr)  #package for data manipulation 
data<- read_excel(f,na="TRUE") #reading file from excel
names(data) #checking column names
dim(data)

######################################################################################################################
#                                   DATA PROCESSING AND SETTING LOGIC                                                #
######################################################################################################################
data[which(is.na(data$Hits)),"Hits"]=0   #Checking if hits is 0 or not and if yes what are the indices
dim(data)
#remove single hit customers
data <- data[-which(data$Hits<=1),]
dim(data)
data$Hits
data$CoV_Sales
dim(data)
# removing CV outliers 
#data<-data[which(data$CoV_Sales>1000),]
dim(data)

# now selecting only the training set without the testing from the original data set. 
train_data<-data %>%
  filter(SheetName != "Test")
# names(train_data)
dim(train_data)
# dim(train_data)[1] +dim(test)[1]

#seeing what are the mean values for column that have missing data, The means are without the missing data. (We use these values and some additional logic to replace 
grps<- train_data%>%
  group_by(Churn)%>%
  summarize(Gm=mean(GMPercent,na.rm=T),Time=mean(AvgTimeDiff,na.rm=T),CV=mean(CoV_Sales,na.rm=T))




#selecting the testing dataset

test<- data%>%
  filter(SheetName=="Test")
dim(test) #checking dimension of testing data

#checking na column in entire data 
na_col<-colnames(test)[colSums(is.na(test))>0]
na_col


#replacing the values without changing the order in the test data. replacing it by values from training data see 'grps' variable

test[which(is.na(test$Hits<=2 & test$AvgTimeDiff)),"AvgTimeDiff"] = grps$Time[2] 
test[which(is.na(test$Hits<=2 &  test$CoV_Sales)),"CoV_Sales"] = grps$CV[2]
test[which(is.na(test$Hits == 2 & test$GMPercent)),"GMPercent"] = grps$Gm[2]
test[which(is.na(test$Hits!=2 & test$AvgTimeDiff)),"AvgTimeDiff"] = grps$Time[1]
test[which(is.na(test$Hits!=2 &  test$CoV_Sales)),"CoV_Sales"] = grps$CV[1]
test[which(is.na(test$GMPercent != 2 & test$GMPercent)),"GMPercent"] = grps$Gm[1]

#checking if test still contains na or not.... should not contain NA anymore
na_col<-colnames(test)[colSums(is.na(test))>0]
#confirmed 
na_col

#checking dimension again 
dim(test)
names(test) #checking names of columns
library(caret) # caret library for preprocessing, model building, confusion matrix
dim(test) # dimensions of test data


#dividing the test set into two equally disjoint sets. We don't need to shuffle the data because objective is to test on a more recent time 

#before dividing checking if any of the customer ID will overlap in any of the future test fold(see code below this). 
#we don't want the customer ID to exist multiple time in the two testing set because we need to test each customer once and only once


# logic, if number of observation is even then just half of the total number of obs else if odd then round 
if (dim(test)[1] %% 2 == 0) {
  l<- dim(test)[1]/2
}else {
  l<- round(0.5*dim(test)[1])
}
l # number of row where splitting will happen 

nrow(test)
# testing_folds$Fold1
intersect(test[1:l,]$customerID,test[l+1:nrow(test),]$customerID) # confirming if disjoint or not 
test_fold1<-test[1:l,] #creating test fold 1 
test_fold2<-test[(l+1):nrow(test),] #creating test fold 2 
#this proves that customerID exist only once in each of the testing folds
#this proves that customers are disjoint in both folds
intersect(test_fold1$customerID,test_fold2$customerID)




# testing_folds$Fold1

#some testing

# intersect(train_fold1[,1],test[testing_folds$Fold1,1])
# intersect(train_fold2[,1],test[testing_folds$Fold2,1])
# 
# setdiff(train_fold2[,1],test[testing_folds$Fold2,1])

#############################################################################################################################

#now we have train data to consider

names(train_data)

#to be sure checking which column in the training set has missing values and replace it based on the reponse value by grouping them, check line 80
na_train<- colnames(train_data)[colSums(is.na(train_data))>0]
na_train
#replacing na values in training data 

train_data[which(is.na(train_data$Churn==0 & train_data$AvgTimeDiff)),"AvgTimeDiff"]= grps$Time[1]
train_data[which(is.na(train_data$Churn==0 & train_data$GMPercent)),"GMPercent"]= grps$Gm[1]
train_data[which(is.na(train_data$Churn==0 & train_data$CoV_Sales)),"CoV_Sales"]= grps$CV[1]
train_data[which(is.na(train_data$Churn==1 & train_data$AvgTimeDiff)),"AvgTimeDiff"]= grps$Time[2]
train_data[which(is.na(train_data$Churn==1 & train_data$GMPercent)),"GMPercent"]= grps$Gm[2]
train_data[which(is.na(train_data$Churn==1 & train_data$CoV_Sales)),"CoV_Sales"]= grps$CV[2]

#checking if na still exist in train data
colnames(train_data)[colSums(is.na(train_data)>0)]






#making index for those customers that are unique only in training data when ompared to entire test data 
only_train_index<-which(train_data$customerID %in% test$customerID=="FALSE")
#making  separate data set for those customers that exist only in training data 
C<-train_data[only_train_index,]
#deleting those customers from original dataframe to make sure we are not duplicating customers
train_data<-train_data[-only_train_index,]


#randomly sampling those customers that are only present in train into either of the two training folds
set.seed(111)
index<-sample(1:nrow(C),size=round(0.5*nrow(C)),replace=F)
#C1_train is training fold 1
C1_train<- C[index,]
#traning fold 2
C2_train<- C[-index,]
# dim(C1_train)
# dim(C2_train)
# dim(train_data)

#two random folds are created
# train_data[train_data$customerID %in% C$customerID=="TRUE",]
# r1<-which(train_data$customerID %in% C$customerID=="TRUE")
# train_data<- train_data[-r1,]

#checking intersection
sum(train_data$customerID %in% C$customerID) # should be 0

#removed customers from original train adta as already placed them in two folds
# train_data[train_data$customerID %in% C$customerID=="TRUE",]

# dim(train_data)


#now checking the remaining customers in original training data with test fold 1

which(train_data$customerID %in% test_fold1$customerID=="TRUE")

#seeing which customers in test fold 1 also present in training data
to_fold2_train<-which(train_data$customerID %in% test_fold1$customerID=="TRUE")
#inputting the customers that are present in test 1 and training in fold 2 of training
C2_train$customerID %in% test_fold1$customerID
C2_train<-rbind(train_data[to_fold2_train,],C2_train)
C2_train$customerID %in% test_fold1$customerID
sum(train_data[to_fold2_train,]$customerID %in% test_fold2$customerID)
#deleting the training values which we just placed in C2_train from original train data
train_data<-train_data[-to_fold2_train,]
dim(train_data)

#repeating process for test set 2 now. objective is to find index of the customers that are overlapping between remaining 
#training data and and test set 2 and taking those customer indexes and placing then=m in fold 1 of training (C1_train)

# checking customers that exist in original train data and test2 now.
train_data$customerID %in% test_fold2$customerID

#index of fold1
to_fold1_train<-which(train_data$customerID %in% test_fold2$customerID=="TRUE")


#binding
C1_train<- rbind(train_data[to_fold1_train,],C1_train)

#proof that Intersection of test fold 1 and Fold 1 train is 0
sum(C1_train$customerID %in% test_fold1$customerID)
#proof that intersection of test fold 2 and train fold 2 are 0
sum(C2_train$customerID %in% test_fold2$customerID)
train_data<-train_data[-to_fold1_train,]
dim(train_data)

#that test fold 1 and fold 2 are disjoint set proof
sum(test_fold2$customerID %in% test_fold1$customerID)
#hence proved. 
intersect(C1_train$customerID,test_fold1$customerID)
intersect(C2_train$customerID,test_fold2$customerID)

# dim(C1_train)
# dim(C2_train)

# in summary we developed a simple yet effectve technique to ensure the following : 
#   1) the customers in testing set are tested only once in each iteration (2) and therefore 
# developed two disjoint sets of test fold1 and fold 2
# 
# 2) then we made sure that customers in training and test set are not overlapping in each iteration
# 
# 3) for example iteration one will be over C1_train and test fold 1 with no overlapping customers 
# 4) second iteration will be over C2_train and test fold 2 with no overlapping customers 
# 
# # ultimate goal of doing this was we need to test on out of period data and not out of sample
# # which in other words is very useful for time series data. need to test on most recent time while 
# # capturing and training over as much data as we can without any overapping customers. 
# # we created two training set and two test set ultiated to conduct a nested 5X2 cross validation where the outer loop will be used for final testing and calcultaing AUC 
# inner loop to tune hyperparamters



# checking imbalance in training and testing data . Might help us to make a judgement in order to which weights to use during our 
#weighted random forest model
prop.table(table(C1_train$Churn)) #proportion of Churn and non Churn in C1_train
table(C2_train$Churn)       
prop.table(table(C2_train$Churn)) # proportion of Churn and non Churn in C2_train
prop.table(table(test_fold1$Churn)) #proportion of Churn and non Churn in fold1 test
prop.table(table(test_fold2$Churn)) #proportion of Churn and non Churn in fold2 test
dim(C2_train)
dim(test_fold2)
dim(C1_train)
dim(test_fold1)


library(randomForest)


# building a simple model initially without any tuning or feature engineering
#justa  abseline model
library(caret)

set.seed(123)

# tControl <- trainControl(
#   method = "repeatedcv", #cross validation
#   #number = 10, #10 folds
# #   #search = "random" #auto hyperparameter selection
# #   repeats = 10,
# #   summaryFunction = twoClassSummary,
# #   classProbs = TRUE,
# #   savePredictions = T)
# trRf <- train(
#   y ~ ., #formulae
#   data = training[,-1], #data without id field
#   method = "rf", # random forest model
#   metric = "Kappa",
#   ntree = 500,
#   trControl = tControl) # train control from previous step.


set.seed(123)

# tControl <- trainControl(
#   method = "repeatedcv", #cross validation
#   #number = 10, #10 folds
#   #search = "random" #auto hyperparameter selection
# )

######################################################################################################################
#                                       DATA FINALIZING AND MODELLING PHASE                                          #
######################################################################################################################



#deleting unwanted column now, such as sheetname and LocationName..... Because I am familiar with data set I know locationNname is not an important factor
names(C1_train)
C1_train<-C1_train %>%
  select(-SheetName,-locationName)
C1_train$Churn <- factor(C1_train$Churn) #converting Churn to a factor or a qualitative rather than charachter 
names(C1_train)

#making different name for response label as it is a requirement for the caret package. 1 and 0 are converted to X1 and X0
C1_train<- C1_train %>% 
  mutate(Churn = factor(Churn,labels = make.names(levels(Churn))))

#repeating the same process for the second training set and testing sets
#for C2
C2_train<-C2_train %>%
  select(-SheetName,-locationName)
C2_train$Churn <- factor(C2_train$Churn)
names(C2_train)

C2_train<- C2_train %>% 
  mutate(Churn = factor(Churn,labels = make.names(levels(Churn))))

#repeating same for test fold 1
test_fold1<- test_fold1%>%
  select(-SheetName,-locationName)
test_fold1$Churn<- factor(test_fold1$Churn)
dim(test_fold1)
dim(C1_train)
test_fold1<- test_fold1 %>% 
  mutate(Churn = factor(Churn,labels = make.names(levels(Churn))))

#for test fold 2


test_fold2<- test_fold2%>%
  select(-SheetName,-locationName)
test_fold2$Churn<- factor(test_fold2$Churn)
dim(test_fold2)

test_fold2<- test_fold2 %>% 
  mutate(Churn = factor(Churn,labels = make.names(levels(Churn))))
dim(test_fold2)
dim(test_fold1)


#initializing lists  for nested CV later
inner_list<-list()
inner_list<- list(C1_train,C2_train)
outer_list<- list(test_fold1,test_fold2)

# tuning_over_inner<- list()
# predictions_outer<- list ()
# prob_pred<- list()
library(MLmetrics)

#converting maret segments into factor 
inner_list[[1]]$marketSegment<- factor(inner_list[[1]]$marketSegment)
inner_list[[2]]$marketSegment<- factor(inner_list[[2]]$marketSegment)
outer_list[[1]]$marketSegment<- factor(outer_list[[1]]$marketSegment)
outer_list[[2]]$marketSegment <- factor(outer_list[[2]]$marketSegment)


#using boruta for feature selection... just to have an idea what the features seems like. WE ALSO do RFE to select features = later on. 

#3  thing to keep in mind is that customers in the training set exist multiple times and have overlaps as we are conducting a multi time slice approach 
#features are calculated for the earliest 9 months and subsequently response labels are calculated three months from that. then we make a new training set by conducting a one month shift 
#we do this until we have exhausted the 2 years worth of data. 
#however the testing set ensures a three month shift rather than a month as we don't want any information leakage between training and testing. 
#in short, in the end, we end up with 10 different training data each with a one month shift after initial 9 months of data used to develop the feature space 
#and testing set with a three month shift. the feature space window is 9 month and response or churn label is three months as we need to predict quarterly churn. 

library(Boruta)
boruta<- Boruta(Churn~., data = outer_list[[1]][,-1],doTrace=2,maxRuns=300)
print(boruta)
plot(boruta,las=2,cex.axis=0.7)
boruta$finalDecision

attStats(boruta)
getNonRejectedFormula(boruta) #taking only those that relevant from boruta 
teb<-TentativeRoughFix(boruta)
teb$finalDecision
teb$ImpHistory

getNonRejectedFormula(teb) #this gives you the formula that goes into training

#code for recursive feature elimination but very ime consuming and computational heavy 
# m_rfe<-list()
# library(rsample)
# set.seed(20140102)
# 
# 
# 
# for (i in 1:2) {
#   rfFuncs$summary <- twoClassSummary
#   subs <- c(1:49)
#   ctrl_rfe <- rfeControl(functions = rfFuncs,method = "cv", number = 5,
#                          verbose=T)
#   
#   
#   m_rfe[[i]] <- rfe(Churn ~., data = inner_list[[i]][,-1],
#                     metric = "ROC",
#                     sizes=subs,
#                     rfeControl = ctrl_rfe)
# }
# 
# m_rfe[[1]]
# m_rfe[[2]]
# m_rfe[[1]]$bestSubset
# 
# predictors(m_rfe[[2]])
# best.set<- predictors(m_rfe[[2]])
# 
# inner_list_best<- list()
# inner_list_best[[1]]<-inner_list[[1]]%>%
#   select(best.set)
# inner_list_best[[2]]<-inner_list[[2]]%>%
#   select(best.set[[2]])
# names(inner_list[[1]])




#initializing other lists 
#update inner lists with best subset and churn value 
# do same for outer lists




ranger_fit<- list()
predict_final<- list()
predict_final_prob<- list()
auc_list<- list()
outer_model<- list()
library(ROCR) #package for ROC curve 
library(pROC) #package for AUC 

outer_list
inner_list[[2]]$Churn
length(outer_list[[2]]$Churn)
#nested cv 
set.seed(121)
#model weights is important and judgment must be used to set a number for rare class. in this case 0.15 for positive class and 0.85 for rare. 
#the number should add to one. the higher the rare class weight the better will be churn accuracy but both accuracy shold reasonably be good for a great model. 
#therefore trial and error or a good balance somewhere is required
#in cases of extreme imbalance higher weights for rare class must be considered. 
for (i in 1:2){
  model_weight<- ifelse(inner_list[[i]]$Churn=="X0",
                        (1/table(inner_list[[i]]$Churn)[1])*0.30,
                        (1/table(inner_list[[i]]$Churn)[2])*0.70)
  
  dim(data)
  #tuning for best value of mtry for AUC metric. #mtry is one of the hyper parameters of RF model
  set.seed(121)
  grid <- expand.grid(
    .mtry = c(3,5,7,14,20,28), #mtry cannot be greater than the number of variables
    .splitrule = "gini",
    .min.node.size = c(5,10,15,20) #tuning for minimum node size, another hyperparameter for the model
  )
  
  
  fitControl <- trainControl(method = "CV",
                             number = 5,
                             verboseIter = TRUE,
                             classProbs = TRUE,summaryFunction=prSummary)
  
  #fitting the model here 
  ranger_fit[[i]] = train(
    Churn ~ Sales + COGS + DaysToPay + GM + Hits + Recency + PurchaseSpan + 
      AvgTimeDiff + Orders_Mean + M2_Sales + M6_Sales + M6_Order + 
      M9_Sales + M9_Order + Q1_Sales + Q3_Sales + Q2_Sales + Q1_Order + 
      Q3_Order + Q2_Order + `#Orders` + DaysLate,
    data = inner_list[[i]][,-1],
    method = 'ranger',
    tuneGrid = grid, #using grid to tune hyperparameter 
    weights = model_weight, #passing model weights
    num.trees=1000, #number of trees. Usually a large number so that the error rate settles down
    importance="permutation", #variable importance using permutation method. for more info read embedded feature selection methods
    trControl = fitControl #controls the flow, such as CV, number of times, probability or not, etc
  )
  
  #this is control for outer loop 
  outerCtrl<- trainControl(classProbs = T,summaryFunction = prSummary,verboseIter = T)
  #grids here are taking values from inner loop and fitting again on entire model. 
  outer_grid<- expand.grid(
    .mtry = c(ranger_fit[[i]]$bestTune$mtry),
    .splitrule = "gini",
    .min.node.size = c(ranger_fit[[i]]$bestTune$min.node.size))
  
  
  outer_model[[i]]<-train(Churn ~ Sales + COGS + DaysToPay + GM + Hits + Recency + PurchaseSpan + 
                            AvgTimeDiff + Orders_Mean + M2_Sales + M6_Sales + M6_Order + 
                            M9_Sales + M9_Order + Q1_Sales + Q3_Sales + Q2_Sales + Q1_Order + 
                            Q3_Order + Q2_Order + `#Orders` + DaysLate , data = inner_list[[i]][,-1],
                          method="ranger",
                          tuneGrid= outer_grid,
                          num.trees = 1500,
                          importance = "permutation",
                          weights = model_weight,
                          trControl = outerCtrl)
  
  # don't forget to initialize the lists
  #predicting churn values 
  predict_final[[i]]<- predict(outer_model[[i]],outer_list[[i]])
  #predicting churn probability values
  predict_final_prob[[i]]<- predict(outer_model[[i]],outer_list[[i]],type="prob")
  #calculating AUC metric
  auc_list[[i]]<- auc(outer_list[[i]]$Churn, predict_final_prob[[i]][,2])
  
}
#######################################################################################################################
#                                              PREDICTIONs PHASE                                                     #
#######################################################################################################################

# v<-predict(ranger_fit[[1]],outer_list[[1]])
# confusionMatrix(v,outer_list[[1]]$Churn)
# auc_list[[1]]
# auc_list[[2]]
library(pROC)
library(caTools)

#confusion matrix for both sets now
confusionMatrix(predict_final[[1]],outer_list[[1]]$Churn)$byClass
confusionMatrix(predict_final[[2]],outer_list[[2]]$Churn)$byClass

#ROC CURVE for 1 and 2 data sets

# for 1st 
xx<-predict_final_prob[[1]]$X0
yy<-outer_list[[1]]$Churn
as.numeric(yy)
colAUC(xx,yy,plotROC = TRUE)

#for 2nd
xx2<-predict_final_prob[[2]]$X0
yy2<-outer_list[[2]]$Churn
as.numeric(yy2)
colAUC(xx2,yy2,plotROC = TRUE)

#binding outer list or the data with it's corresponding predictions and probability values
one<-cbind(outer_list[[1]],predicted_churn=predict_final[[1]],prob=predict_final_prob[[1]])
#same for two 
two<- cbind(outer_list[[2]],predicted_churn=predict_final[[2]],prob=predict_final_prob[[2]])

#combining one and two to get complet data. think of adding the CF matrix of one and two.
#we can produce one single CF matrix now

final<-rbind(one,two)

confusionMatrix(final$predicted_churn,final$Churn)$table
confusionMatrix(final$predicted_churn,final$Churn)$byClass
# filtering 
final%>%
  filter(Churn==predicted_churn & Sales> 30000 & Churn == "X1")%>%
  summarize(sum(Sales))
final%>%
  filter(Churn==predicted_churn  & Churn == "X1")%>%
  summarize(sum(Sales))
final%>%
  filter(Churn!=predicted_churn & predicted_churn =="X0")%>%
  summarize(sum(Sales))

#averaging the mean importance of variables
one_imp
one_imp<-data.frame(importance1 = outer_model[[1]]$finalModel$variable.importance)
two_imp<-data.frame(importance2 = outer_model[[2]]$finalModel$variable.importance)

#final importance table
importance<-cbind(one_imp,two_imp)
final_importance<-importance%>%
  mutate(mean_importance = (importance1+importance2)/2)%>%
  arrange(desc(mean_importance))%>%
  select(mean_importance)
final_importance
class(final_importance)





#plot to show the overall importance
ggplot(data=data.frame(order(final_importance,decreasing=TRUE)),aes(final_importance$mean_importance,factor(rownames(final_importance)),fill=factor(rownames(final_importance))))+geom_col()
names(outer_model[[1]])
names(inner_list[[1]])
final_importance%>%
  mutate(perc=mean_importance/sum(mean_importance))
plot(varImp(outer_model[[1]]))
plot(varImp(outer_model[[2]]))



#complete data with predictions file. 
#writing out the final result
write.csv(final,file="complete_test_h1.csv")

#variable importance file
# writing the importance results as a csv
write.csv(final_importance,file="importance_h1.csv")

final_importance



outer_model[[1]]$finalModel
n<-predict(outer_model[[1]],outer_list[[1]])
confusionMatrix(n,outer_list[[1]]$Churn)$byClass
outer_model[[2]]
n
max(inner_list[[1]]$LinePenetration)
inner_list[[1]]$`#Categories`


# filtering to make the dollar value confusion matrix

final%>%
  filter(Churn==predicted_churn & Churn == "X1")%>%
  summarize(sum(Sales))
final%>%
  filter(Churn=="X1")%>%
  summarize(sum(Sales))

final%>%
  filter(Churn==predicted_churn & Churn =="X0")%>%
  summarize(sum(Sales))

final%>%
  filter(Churn!= predicted_churn & predicted_churn == "X1")%>%
  summarize(sum(Sales))

final%>%
  filter(Churn=="X0")%>%
  summarize(sum(Sales))
final%>%
  filter(Churn!= predicted_churn & predicted_churn == "X0")%>%
  summarize(sum(Sales))
final%>%
  summarize(sum(Sales))



dim(final)
result<-final%>%
  select(customerID,prob.X0,prob.X1,predicted_churn,Churn)

#result file 
#writing another file only for predicted probs , predicted churn label, true churn label and cust ID
write.csv(result,file="result_Company_h1.csv")
# Fn<-final%>%
#   filter(Churn!= predicted_churn & predicted_churn == "X0")
# Fn_1<-final%>%
#   filter(Churn!= predicted_churn & predicted_churn == "X0")
# 
# Fn_2<-final%>%
#   filter(Churn!= predicted_churn & predicted_churn == "X1")
# 
# 
# 
# Fn<-rbind(Fn_1,Fn_2)
# 
# write.csv(Fn,"fn_h1.csv")




rate<-final %>%
  group_by(marketSegment)%>%
  count(pp=predicted_churn)%>%
  ungroup()%>%
  mutate(s=sum(n))%>%
  group_by(marketSegment,pp)%>%
  filter(pp=="X1")%>%
  summarize(rate=(n/s)*100)%>%
  arrange(desc(rate))%>%
  ungroup()


rate<-final %>%
  group_by(marketSegment)%>%
  count(pp=Churn)%>%
  ungroup()%>%
  mutate(s=sum(n))%>%
  group_by(marketSegment,pp)%>%
  filter(pp=="X1")%>%
  summarize(rate=(n/s)*100)%>%
  arrange(desc(rate))%>%
  ungroup()
rate
#rate file
write.csv(rate,file="rate_Company_2.csv")

final%>%
  filter(predicted_churn=="X1")%>%
  summarize(sum(Sales))

confusionMatrix(final$predicted_churn,final$Churn)$byclass

##############################################################################################################
#                                   EXTRA STUFF - NOT IMPORTANT                                              #
##############################################################################################################
# str(FN)
# str(Fn)  
# 
# library(factoextra)
# str(Fn)
# cluster_data<- Fn%>%
#   select(-Churn,-marketSegment,-predicted_churn)
# 
# clus_1<-Fn%>%
#   select(-Churn,-marketSegment,-predicted_churn,-prob.X0,-prob.X1)
# names(clus_1)
# 
# 
# str(cluster_data)
# 
# pca_1<-prcomp(cluster_data[,-1],scale=T,center=T)
# 
# summary(pca_1)
# rotation_table<-data.frame(pca_1$x)
# plot(rotation_table$PC1,rotation_table$PC2)     
# 
# library(ggbiplot)
# ggbiplot()
# 
# 
# 
# ls<- data_original %>%
#   select(CustomerFinalRank)
# class(ls)
# length(ls)
# dim(data)
# ls<-ls[-c(1097,452,458,388),]
# 
# class(ls)
# count(ls)
# 
# data_original<-data_original[-c(1097,452,458,388,861,1559,16),]
# 
# Fn$Churn
# 
# 
# #zoomed out view 
# ggbiplot(pca_1,labels.size = 2,ellipse = T,groups=factor(Fn$Churn),choices=c(1,2))+xlim(-2,5)+ylim(-5,2)+theme_classic()
# 
# #zoomed in
# ggbiplot(pca_1,ellipse = T,groups=factor(Fn$Churn),labels=rownames(clus_1),choices=c(1,2))+xlim(-2,5)+ylim(-2,2)+theme_classic()
# #cross check wether the customer has actually churned or not
# ggbiplot(pca_1,ellipse = T,groups=factor(ls$CustomerFinalRank),labels=rownames(pca_data),choices=c(1,2))+xlim(-2,5)+ylim(-2,2)+theme_classic()
# 
# data[199,]
# ggbiplot(pca_1,ellipse = T,groups=factor(data_original$CustomerFinalRank),choices=c(1,2),var.axes=T,varname.adjust = 4,varname.size = 5,alpha=0.6,var.scale=1)+xlim(-2,5)+ylim(-2,2)+theme_dark()
# 
# 
# 
# y<- data$CustomerFinalRank
# x<- pca_1$x[,1:10]
# clus_1<- scale(clus_1[,-1],center = T)
# pca_1
# library(factoextra)
# set.seed(1)
# #We are determining the optimal cluster using silhoutte score for kmeans
# #Silhouette score ranges fron 0 to 1, with 1 being the best
# #For more on Silhouette score: https://vitalflux.com/kmeans-silhouette-score-explained-with-python-example/
# clus<-fviz_nbclust(clus_1[,-1], kmeans, method='silhouette', k.max = 25)
# clus
# set.seed(11)
# km.out<-kmeans(cluster_data[,-1],4,nstart=50)
# summary(km.out)
# km.out$cluster
# clus_res<-km.out$centers
# write.csv(clus_res,"res.csv")
# km.out$cluster
# aggregate(cluster_data,by=list(cluster=km.out$cluster),mean)
# class(x)
# dd<- cbind(cluster_data,cluster=km.out$cluster)
# dd <-data.frame(dd)
# dd
# 
# ggplot(rotation_table,aes(rotation_table$PC1,rotation_table$PC2,col=factor(dd$cluster)))+geom_point(aes(shape=factor(Fn$Churn)))
# plot(x[,1:2],col=(km.out$cluster+1),pch=20,cex=2)
# 
# dd<-cbind(dd,Fn$customerID)
# write.csv(dd,"cluster_43.csv")
# sum(Fn[which(dd$cluster==1),]$Sales)
# sum(Fn[which(dd$cluster==2),]$Sales)
# sum(Fn[which(dd$cluster==3),]$Sales)
# sum(Fn[which(dd$cluster==4),]$Sales)
# sum(Fn[which(dd$cluster==5),]$Sales)
# sum(Fn[which(dd$cluster==6),]$Sales)
# dd<-cbind(dd,Churn =Fn$Churn,Pred_Churn =Fn$predicted_churn)
# dd%>%
#   arrange(desc(cluster))%>%
#   filter(Churn == "X1")
# 
# 
# fviz_cluster(km.out,x[,1:2],repel = TRUE)
# fviz_cluster(km.out,x[,1:2],geom="text")
# fviz_cluster(km.out,x[,1:2],geom="point")
# count(km.out$cluster)
# count(dd$cluster)
# count(ls)
# names(data_ini)
# Fn$Sales
# 
# 
# 
# dim(data %>%
#   filter(marketSegment=="GROCERY"))
# dim(final)
# 
# 
# 
# 
# 
# 
# final%>%
#   filter(marketSegment == "GROCERY" & Churn != predicted_churn & prob.X0 >0.60)
