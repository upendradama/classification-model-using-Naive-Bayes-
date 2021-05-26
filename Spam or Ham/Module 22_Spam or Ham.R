# Naive Bayes
#Problem Statement :- Classifying Spam or Ham
#Loading the dataset
sms <- read.csv("~/desktop/Digi 360/Module 22/Datasets-6/sms_raw_NB.csv", encoding = 'latin1')
head(sms)
attach(sms)
attach(sms)
#Data Cleansing and Processing
# Applying Factor for text
sms$type <- factor(sms$type)
head(sms)
# Examine the structure of varibales

str(sms$type)
table(sms$type)

# Building Corpus using Text Mining Package "tm"
install.packages("tm")
library(tm)
sms_corpus <- Corpus(VectorSource(sms$text))
# Converting into UTF-8 Characters
sms_corpus <- tm_map(sms_corpus, function(x) iconv(enc2utf8(x), sub='byte'))

# clean up the corpus using tm_map()
corpus_clean <- tm_map(sms_corpus, tolower)
corpus_clean <- tm_map(corpus_clean, removeNumbers)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, stripWhitespace)

# create a document-term sparse matrix
sms_dtm <- DocumentTermMatrix(corpus_clean)
sms_dtm

# Splitting Raw dataset
library(caret)
train_index <- createDataPartition(sms$type, p=0.75, list=FALSE)

sms_train <- sms[train_index,]
sms_test <- sms[-train_index,]
corpus_clean_train <- corpus_clean[train_index]
corpus_clean_test <- corpus_clean[-train_index]
sms_dtm_train <- sms_dtm[train_index,]
sms_dtm_test <- sms_dtm[-train_index,]

# check that the proportion of spam is similar
prop.table(table(sms_train$type))
prop.table(table(sms_test$type))

# indicator features for frequent words
# dictionary of words which are used more than 5 times

sms_dict <- findFreqTerms(sms_dtm_train, 5)

sms_train1 <- DocumentTermMatrix(corpus_clean_train, list(dictionary = sms_dict))
sms_test1  <- DocumentTermMatrix(corpus_clean_test, list(dictionary = sms_dict))

# convert counts to a factor
# custom function: if a word is used more than 0 times then mention 1 else mention 0
convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
}

# apply() convert_counts() to columns of train/test data
# Margin = 2 is for columns
# Margin = 1 is for rows

sms_train1 <- apply(sms_train1, MARGIN = 2, convert_counts)
sms_test1  <- apply(sms_test1, MARGIN = 2, convert_counts)

#Building the Model

install.packages("e1071", dependencies=TRUE)
library(e1071)

# Training a model on the data ----

library(e1071)
sms_classifier <- naiveBayes(sms_train1, sms_train$type)
sms_classifier

#Model Evaluation

# Predicting on test
sms_test_pred <- predict(sms_classifier, sms_test1)

table(sms_test_pred)
prop.table(table(sms_test_pred))

install.packages("gmodels")
library(gmodels)
CrossTable(sms_test_pred, sms_test$type,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

#Conclusion:- We can see that 2 ham messages have been predicted as spam. So we can conclude that our model is good. 
