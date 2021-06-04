#Capstone Project "No-Show appointment" 

rm(list = ls()) # clear global environment
graphics.off() # close all graphics
gc()

#Data Preparation Phase 1 

#improting data 
noshow <- read.csv("Noshow_data.csv", header = TRUE)

#explore data 

str(noshow)
summary (noshow)

head(noshow)

#Gender, AppointmentDay and Show.up are Categorical variables, 
#so we need to encode them 


#install.packages('fastDummies')
library('fastDummies')

noshow1 <- dummy_cols(noshow, select_columns = c('Gender', 'Show.up', 'AppointmentDay', 'Month'), remove_selected_columns = TRUE)

noshow1 = subset(noshow1, select= -c(Gender_M, Show.up_no, PatientId, AppointmentID, AppointmentDay_Saturday))

str(noshow1)

############################################# try to remove month at all because there is no information for the whole year 

#zero and near zero varience Predictors
library(caret)
Near_Zero_Var <- nearZeroVar(
  noshow1,
  freqCut = 95/5,
  uniqueCut = 10,
  saveMetrics = TRUE)

#As we can see there were 3 near zero variance predictors identified and they 
#include Alcoholism, Handicap and AppointmentDay_Saturday

library(dplyr)

# visualising Appointment Day for the Near Zero varience step 
AppointmentDay.freq <- noshow  %>%
  group_by(AppointmentDay) %>%
  summarize(count=n()) %>%
  arrange(desc(count))

ggplot(AppointmentDay.freq, aes(x=reorder (AppointmentDay,count), count, fill=count)) + 
  geom_bar(stat="identity") +
  xlab("AppointmentDay_Saturday") + 
  ylab("Number of Appointments") + 
  ggtitle("Appointment frequency depending on AppointmentDay_Saturday") + 
  theme_classic() +
  geom_text(aes(label=count), vjust=-0.5) +
  scale_fill_gradient(low ="#ee4000", high ="#8968cd")

# There were almost no appointments made in Saturday
# In total there were only 19 appointments made in Saturday (whoch is 0.03% of total number of appointments) and 7 of them were noshow 
# So I decided to delete these observations. 

#Now we need to have a look at Handicap

#visualising Handicap 
Handicap.freq <- noshow  %>%
  group_by(Handicap) %>%
  summarize(count=n()) %>%
  arrange(desc(count))

ggplot(Handicap.freq, aes(Handicap, count, fill=count)) + 
  geom_bar(stat="identity") +
  xlab("Handicap") + 
  ylab("Number of Appointments") + 
  ggtitle("Appointment frequency depending on Handicap") + 
  theme_classic() +
  geom_text(aes(label=count), vjust=-0.5) +
  scale_fill_gradient(low ="#ee4000", high ="#8968cd")


#lets try to combine 1,2 and 3 together 

noshow1$Handicap <- as.character(noshow1$Handicap)
noshow1$Handicap[noshow1$Handicap == "2"] <- "1"
noshow1$Handicap[noshow1$Handicap == "3"] <- "1"
str(noshow1)

noshow1$Handicap <- as.integer(noshow1$Handicap)
str(noshow1)


Handicap.freq <- noshow1  %>%
  group_by(Handicap) %>%
  summarize(count=n()) %>%
  arrange(desc(count))

ggplot(Handicap.freq, aes(Handicap, count, fill=count)) + 
  geom_bar(stat="identity") +
  xlab("Handicap") + 
  ylab("Number of Appointments") + 
  ggtitle("Appointment frequency depending on Handicap") + 
  theme_classic() +
  geom_text(aes(label=count), vjust=-0.5) +
  scale_fill_gradient(low ="#ee4000", high ="#8968cd")

#Handicap cases where handicap was =1,2 and 3 did not account for 5% oa all handicap cases
# deleting handicap 1,2,3

#noshow1 = filter(noshow1, Handicap != "1")
# feature selection - have a look later 

# Looking into Alcoholism data 


#vizualising alcoholism 
Alcoholism.freq <- noshow  %>%
  group_by(Alcoholism) %>%
  summarize(count=n()) %>%
  arrange(desc(count))

ggplot(Alcoholism.freq, aes(Alcoholism, count, fill=count)) + 
  geom_bar(stat="identity") +
  xlab("Alcojolism") + 
  ylab("Number of Appointments") + 
  ggtitle("Appointment frequency depending on Alcoholism") + 
  theme_classic() +
  geom_text(aes(label=count), vjust=-0.5) +
  scale_fill_gradient(low ="#ee4000", high ="#8968cd")


#As we can see number of observations is unevenly distributed between people 
#who suffer from alcoholism and those who are not, but deleting this data might
#not be the best solution in this case, lets see if alcoholism will stay as 
#important variable after feature selection

#noshow1 = filter(noshow1, Alcoholism != "1")
# feature selection 

Near_Zero_Var <- nearZeroVar( 
  noshow1,
  freqCut = 95/5,
  uniqueCut = 10,
  saveMetrics = TRUE)

#Still for a lot of values the frequency ration is above 1

###################### Checking for Missing values ##################

sum(is.na(noshow1))
# no missing values were dectected. 

######################### Detecting Outliers ##########################

summary(noshow1)
str(noshow1)

#Using Cook's method to detect outliers 
mod <- lm(Show.up_yes ~ ., data=noshow1)
cooksd <- cooks.distance(mod)

plot(cooksd, pch="*", cex=2, main="Influential Obs by Cooks distance")  # plot cook's distance
abline(h = 4/nrow(noshow1), col="red")  # add cutoff line
text(x=1:length(cooksd)+1, y=cooksd, labels=ifelse(cooksd>4/nrow(noshow1),names(cooksd),""), ) # add labels


########################## Removing Outliers ################################

# influential row numbers
influential <- as.numeric(names(cooksd)[(cooksd > (4/nrow(noshow1)))])

# Alternatively, you can try to remove the top x outliers to have a look
# top_x_outlier <- #any number that you want to specify 
# influential <- as.numeric(names(sort(cooksd, decreasing = TRUE)[1:top_x_outlier]))

noshow1 <- noshow1[-influential, ]
str(noshow1)

#subset(noshow1, Age<0)

############################ Modeling ############################

# balance split with 75% in train and 25% of data in test 

library(caret)
set.seed(3456)
trainIndex <- createDataPartition(noshow1$Show.up_yes, p = .8, 
                                  list = FALSE, 
                                  times = 1)
head(trainIndex)

noshow1_train <- noshow1[ trainIndex,]
noshow1_test  <- noshow1[-trainIndex,]


######################################imbalanced Data - SMOT ########################

#Balancing train dataset using DBSMOTE 

ap <- available.packages()

#install.packages("smotefamily")

library(smotefamily)

set.seed(10)

noshow1_train_balanced <-ADAS(noshow1_train[,-12], noshow1_train[,12], K=5) 

str(noshow1_train_balanced[["data"]])

#Class variable shows to be a Character, have to change to numeric for futher analysis 
noshow1_train_balanced[["data"]]$class<- as.numeric(noshow1_train_balanced[["data"]]$class)

str(noshow1_train_balanced[["data"]])



############################Features by Importance ###########################
set.seed(7)

#install.packages("mlbench")
library(mlbench)
library(caret)

# prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=5)
# train the model
model <- train(class ~., data=noshow1_train_balanced[["data"]], method="glm", preProcess="scale", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)


str(noshow1_train_balanced[["data"]])
str(noshow1_test)
str(noshow1_train)

noshow1_train_balanced[["data"]] = subset(noshow1_train_balanced[["data"]], select= -c(AppointmentDay_Thursday,AppointmentDay_Friday, Calling_time..hour.in.a.day., Age,AppointmentDay_Wednesday, Month_6))
noshow1_test = subset(noshow1_test, select= -c(AppointmentDay_Thursday, AppointmentDay_Friday, Calling_time..hour.in.a.day., Age,AppointmentDay_Wednesday, Month_6))
noshow1_train = subset(noshow1_train, select= -c(AppointmentDay_Thursday, AppointmentDay_Friday, Calling_time..hour.in.a.day., Age,AppointmentDay_Wednesday, Month_6))


############################# Feature Selection LASSo ########################

#install.packages("glmnet")

library(data.table)    # provides enhanced data.frame
library(ggplot2) # plotting
library(Matrix)
library(glmnet)

x = model.matrix(class ~.,noshow1_train_balanced[["data"]]) # matrix of predictors
y = noshow1_train_balanced[["data"]]$class             # vector y values

set.seed(12)# replicate  results

lasso_model <- cv.glmnet(x, y, alpha=1,family = "binomial")      # alpha=1 is lasso
best_lambda_lasso <- lasso_model$lambda.1se  # largest lambda in 1 SE
lasso_coef <- lasso_model$glmnet.fit$beta[,  # retrieve coefficients
                                          lasso_model$glmnet.fit$lambda  # at lambda.1se
                                          == best_lambda_lasso]
coef_l = data.table(lasso = lasso_coef)      # build table
coef_l[, feature := names(lasso_coef)]       # add feature names
to_plot_r = melt(coef_l                      # label table
                 , id.vars='feature'
                 , variable.name = 'model'
                 , value.name = 'coefficient')
ggplot(data=to_plot_r,                       # plot coefficients
       aes(x=feature, y=coefficient, fill=model)) +
  coord_flip() +         
  geom_bar(stat='identity', fill='brown4', color='blue') +
  facet_wrap(~ model) + guides(fill=FALSE)



#################################### GLM Regression ##################################

noshow1_train_balanced[["data"]]$class<- as.factor(noshow1_train_balanced[["data"]]$class)

str(noshow1_train_balanced[["data"]])

str(noshow1_test)

noshow1_test$Show.up_yes<- as.factor(noshow1_test$Show.up_yes)

str(noshow1_test)

mylogit <- glm(class ~., data = noshow1_train_balanced[["data"]], family = "binomial")


summary(mylogit)

probability <- predict(mylogit, newdata = noshow1_test, type = "response")

test_predict1 <- rep( 0, 13827)

test_predict1[probability >=0.5] = 2

test_predict1[probability < 0.5] = 1

test_predict1<- as.factor(test_predict1)

levels(noshow1_test$Show.up_yes) <- levels(test_predict1)

confusionMatrix(test_predict1, noshow1_test$Show.up_yes)

library(pROC)
AUC <- auc(as.numeric(test_predict1), as.numeric(noshow1_test$Show.up_yes))
AUC
gmean = sqrt(0.6516*0.7279)
gmean




########################### Randm Forest ###############################

#install.packages("randomForest")
library(randomForest)


random_forest <- randomForest(class ~., data = noshow1_train_balanced[["data"]], mtry = 5, importnace  = T)
predict_random_forest <- predict(random_forest, newdata = noshow1_test[-10])

levels(noshow1_test$Show.up_yes) <- levels(predict_random_forest)

confusionMatrix(predict_random_forest,noshow1_test$Show.up_yes)

precision <-posPredValue(predict_random_forest, noshow1_test$Show.up_yes, positive = "1")
precision 

AUC <- auc(as.numeric(predict_random_forest), as.numeric(noshow1_test$Show.up_yes))
AUC
gmean = sqrt(0.042827*0.988794)
gmean

##############Desicion Tree#########################

#install.packages("rpart.plot")	
library(rpart.plot)
library(rpart)
control <- rpart.control(minsplit = 4,
                         minbucket = round(5 / 3),
                         maxdepth = 3,
                         cp = 0)

decision_tree <- rpart(class ~., data=noshow1_train_balanced[["data"]], method='class')

rpart.plot(decision_tree, extra = 110)

predict_decision_tree <-predict(decision_tree, noshow1_test, type = 'class')

confusionMatrix(noshow1_test$Show.up_yes, predict_decision_tree)

precision <-posPredValue(predict_decision_tree, noshow1_test$Show.up_yes, positive = "1")
precision 

AUC <- auc(as.numeric(predict_decision_tree), as.numeric(noshow1_test$Show.up_yes))
AUC
gmean = sqrt(0.3011*0.9046)
gmean



########################### Random Forest_Tuned ###############################

#install.packages("randomForest")
library(randomForest)
random_forest <- randomForest(class ~., data = noshow1_train_balanced[["data"]], sampsize=.1*length(y), ntree=5000, maxnodes=24)

predict_random_forest <- predict(random_forest, newdata = noshow1_test[-10])

levels(noshow1_test$Show.up_yes) <- levels(predict_random_forest)

confusionMatrix(predict_random_forest,noshow1_test$Show.up_yes)


str(noshow1_train_balanced[["data"]])

precision <-posPredValue(predict_random_forest, noshow1_test$Show.up_yes, positive = "1")
precision 

AUC <- auc(as.numeric(predict_random_forest), as.numeric(noshow1_test$Show.up_yes))
AUC
gmean = sqrt(0.640*0.692)
gmean



#This model overfits, so recomendatin for further research: try to modify the model 
#by parameters and resolve an issue of overfitting to use Random forest in research
predict_random_forest <- predict(random_forest, newdata = noshow1_train_balanced[["data"]][-14])

levels(noshow1_train_balanced[["data"]]$class) <- levels(predict_random_forest)

confusionMatrix(predict_random_forest,noshow1_train_balanced[["data"]]$class)

precision <-posPredValue(predict_random_forest, noshow1_train_balanced[["data"]]$class, positive = "1")
precision 

AUC <- auc(as.numeric(predict_random_forest), as.numeric(noshow1_train_balanced[["data"]]$class))
AUC
gmean = sqrt(0.8456*0.7638)
gmean





system.time({ mylogit <- glm(class ~., data = noshow1_train_balanced[["data"]], family = "binomial") })
system.time({ random_forest <- randomForest(class ~., data = noshow1_train_balanced[["data"]], sampsize=.1*length(y), ntree=5000, maxnodes=24)})