# packages
library(tidyverse)
library(ggplot2)
library(randomForest)
library(forcats)
library(stringr)
library(caTools)

# read data
train <- read_csv("train.csv")
test <- read_csv("test.csv")

# Combine the train and test datasets for easier analysis
train$set <- "train"
test$set  <- "test"
test$Survived <- NA
full <- rbind(train, test)

#Data wrangling

str(full)
sapply(full, function(x) length(unique(x)))   # No: of unique values

missing_values <- full%>% summarise_all(funs(sum(is.na(.))/n()))
missing_values <- gather(missing_values, key="feature", value="missing_pct")
missing_values %>% 
  ggplot(aes(x=reorder(feature,-missing_pct),y=missing_pct)) +
  geom_bar(stat="identity",fill="red")+
  theme_bw() + coord_flip()

full$Age <- ifelse(is.na(full$Age), mean(full$Age, na.rm = TRUE), full$Age)
full <- full %>% mutate(Age_Group = case_when(Age < 13 ~ "Age.0012", 
                                               Age >= 13 & Age < 18 ~ "Age.1317",
                                               Age >= 18 & Age < 60 ~ "Age.1859",
                                               Age >= 60 ~ "Age.60Ov"))

full$Embarked <- replace(full$Embarked, which(is.na(full$Embarked)), "S")



Designation <- strsplit(full$Name, split = "[,.]")          # split the string into 3 string parts
full$title <- sapply(Designation, function(x) x[2])   # extracting the middle part of the string
unique(full$title)
full$title <- str_trim(full$title, side = "left") # trim the white space from the left side
table(full$title)

full$title[full$title == 'Capt'|full$title == 'Sir'|full$title == 'Col'|full$title == 'Don'|full$title == 'Major'|full$title == 'Jonkheer'|full$title == 'Rev'] <- 'Mr' 
full$title[full$title == 'Dona'|full$title == 'the Countess'|full$title == 'Mme'|full$title == 'Lady'] <- 'Mrs'
full$title[full$title == 'Mlle'|full$title == 'Ms'] <- 'Miss'
full$title[full$title == 'Dr'& full$Sex == 'male'] <- 'Mr'
full$title[full$title == 'Dr'& full$Sex == 'female'] <- 'Mrs'

full$Cabin <- str_sub(full$Cabin, 1, 1)
full$Cabin[which(is.na(full$Cabin))] <- "Unknown"
full$Fare[is.na(full$Fare)] <- mean(full$Fare, na.rm = TRUE)

full$fam_size <- full$SibSp + full$Parch + 1
full$FamilySized[full$fam_size == 1] <- 'Single' 
full$FamilySized[which(full$fam_size >= 2 & full$fam_size < 5 )] <- 'Small' 
full$FamilySized[full$fam_size >= 5] <- 'Big' 
full$FamilySized=as.factor(full$FamilySized)


##### separating Training data from the full data and splitting it further into training and test data
feature_data <- full %>% filter(set == "train") %>% select(Survived, Pclass, Sex, Age_Group, Fare, Cabin, Embarked, title, FamilySized)
feature_data$Survived <- factor(feature_data$Survived, levels = c(0,1), labels = c("No","Yes"))
feature_data$Pclass <- factor(feature_data$Pclass, levels = c(1,2,3), labels = c("Class1", "Class2", "Class3"))
feature_data$Sex <- as.factor((feature_data$Sex))
feature_data$Cabin <- as.factor((feature_data$Cabin))
feature_data$Embarked <- as.factor((feature_data$Embarked))
feature_data$title <- as.factor((feature_data$title))
feature_data$Age_Group <- as.factor((feature_data$Age_Group))
feature_data$FamilySized <- as.factor((feature_data$FamilySized))


set.seed(123)
split = sample.split(feature_data$Survived, SplitRatio = 0.8)
training_set = subset(feature_data, split == TRUE)
test_set = subset(feature_data, split == FALSE)

####check the proprtion of Survival rate in orginal training data, current training and testing data

round(prop.table(table(train$Survived)*100),digits = 1)

#### Machine learning -- RandomForest
formula1 = Survived ~ Pclass + Sex + Fare + title
formula2 = Survived ~ Pclass + Sex + Fare + Cabin + title
formula3 = Survived ~ .
RFmodel1 <- randomForest(formula1,data = training_set,ntree = 1000, importance = TRUE)
RFmodel2 <- randomForest(formula2,data = training_set,ntree = 1000, importance = TRUE)
RFmodel3 <- randomForest(formula3,data = training_set,ntree = 1000, importance = TRUE)
varImpPlot(RFmodel3)


# Prediction on split test data

test_set$pred <- predict(RFmodel1, test_set)
test_set$pred2 <- predict(RFmodel2, test_set)
test_set$pred3<- predict(RFmodel3, test_set)
confusionMatrix(test_set$Survived, test_set$pred3)
table(test_set$Survived, test_set$pred3)

##### Test data for testing
feature_data1 <- full %>% filter(set == "test") %>% select(Survived, Pclass, Sex, Age_Group, Fare, Cabin, Embarked, title, FamilySized)
feature_data1$Survived <- factor(feature_data1$Survived, levels = c(0,1), labels = c("No","Yes"))
feature_data1$Pclass <- factor(feature_data1$Pclass, levels = c(1,2,3), labels = c("Class1", "Class2", "Class3"))
feature_data1$Sex <- as.factor((feature_data1$Sex))
feature_data1$Cabin <- as.factor((feature_data1$Cabin))
feature_data1$Embarked <- as.factor((feature_data1$Embarked))
feature_data1$title <- as.factor((feature_data1$title))
feature_data1$Age_Group <- as.factor((feature_data1$Age_Group))
feature_data1$FamilySized <- as.factor((feature_data1$FamilySized))

finalTest <- feature_data1
levels(finalTest$Cabin) <- levels(feature_data$Cabin)

Survival1 <- predict(RFmodel1, finalTest)
Survival2 <- predict(RFmodel2, finalTest)
Survival3 <- predict(RFmodel3, finalTest)

Pred_table <- test %>% select(PassengerId)
Pred_table$Survived <- Survival3
head(Pred_table)
colnames(Pred_table) <- c("PassengerId", "Survived")

write_csv(Pred_table, "PredictionTitanic")
######### RFmodel3 gives 80% correct prediction....


#Applying Grid search for hyperparameter tuning
library(caret)
myControl = trainControl(method = 'cv', number = 10, 
                         summaryFunction = twoClassSummary, 
                         allowParallel = TRUE,
                         classProbs = TRUE, verboseIter = TRUE)
model_rf <- train(formula2, data = feature_data, method = 'rf', 
                  tuneGrid = expand.grid(mtry = seq(1,10,by=1)),
                  ntrees= 500,trControl = myControl, metric = "ROC")

# prediction on the final test data 
Survived <- predict(model_rf, finalTest, type = 'raw')
Survived <- ifelse(Survived == "Yes", 1, 0)
Final_Pred <- data.frame(PassengerId = test$PassengerId, Survived)

write_csv(Final_Pred, "PredictionTitanic.csv")

#### Trying Naive_bayern Classification Technique

library(naivebayes)
train_dataNB <- feature_data %>%
                select(-Fare)
test_dataNB <- feature_data1 %>%
                select(-Fare)
# Splitting train_dataNB into train and test data set

set.seed(500)
split = sample.split(train_dataNB$Survived, SplitRatio = 0.8)
training_NB = subset(train_dataNB, split == TRUE)
test_NB = subset(train_dataNB, split == FALSE)

#### modelling training_NB using naive_bayes

modelNB <- naive_bayes(Survived ~ Pclass + Sex + title , training_NB)

SurvivedNB <- predict(modelNB, test_NB)
mean(test_NB$Survived == SurvivedNB)

final_pred <- predict(modelNB, test_dataNB)

Pred_table$Survived <- final_pred

write_csv(Pred_table, "NB_prediction_titanic")
## Naive Bayes a prediction accuracy of 77%



# Modelling using XGBoost
# One Hot Encoding of the dependent variables
library(vtreat)
vars <- colnames(feature_data[-1])
treatplan <- designTreatmentsZ(feature_data, vars, verbose = FALSE)
(newvars <- treatplan$scoreFrame %>%
            filter(code %in% c("clean", "lev")) %>%
            select(varName))
newvars <- newvars$varName
feature_data$Survived = factor(feature_data$Survived, levels = c("No", "Yes"), labels = c(0,1))
feature_data1$Survived = factor(feature_data1$Survived, levels = c("No", "Yes"), labels = c(0,1))
feature_data$Survived <- as.numeric(feature_data$Survived) - 1
feature_data1$Survived <- as.numeric(feature_data1$Survived) - 1
feature_data_treat <- prepare(treatplan, feature_data, varRestriction = newvars)
finaltest_treat <- prepare(treatplan, feature_data1, varRestriction = newvars)

# Finding the appropriate number of trees  # dependent variable has to be numeric for xgb to work
cv <- xgb.cv(data = as.matrix(feature_data_treat), 
             label = feature_data$Survived,
             nrounds = 100,
             nfold = 5,
             objective = "binary:logistic",
             eta = 0.3,
             max_depth = 6,
             early_stopping_rounds = 10,
             verbose = 0    # silent
)

elog <- cv$evaluation_log # get the evaluation log
# Determine and print how many trees minimize training and test error
elog %>% 
  summarize(ntrees.train = which.min((train_error_mean)),   # find the index of min(train_rmse_mean)
            ntrees.test  = which.min((test_error_mean))) 
ntree = cv$best_iteration
model_xgb <- xgboost(data = as.matrix(feature_data_treat), # training data as matrix
                     label = feature_data$Survived,  # column of outcomes
                     nrounds = ntree,       # number of trees to build
                     objective = "binary:logistic", # objective
                     eta = 0.3,
                     depth = 6,
                     verbose = 0  # silent
)

Survived <- predict(model_xgb, as.matrix(feature_data1_treat), type = 'response')
Survived <- ifelse(Survived >0.5, 1, 0)
Final_Pred <- data.frame(PassengerId = test$PassengerId, Survived)
write_csv(Final_Pred, "PredXGBoost.csv" )
