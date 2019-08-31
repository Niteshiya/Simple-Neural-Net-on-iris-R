#load libraries
library(neuralnet)
library(nnet)
library(keras)
library(NeuralNetTools)
install_keras()
#import data frame
data <- as.data.frame(iris)
head(data)
str(data)
#data wrangling and cleaning
species <- data$Species %>% levels()
species
data$Species <- as.numeric(data$Species)
head(data)
#---------------normalise
mat <- as.matrix(data)
dim(mat)
nor_mat <- mat 
nor_mat[,1:4] <- normalize(nor_mat[,1:4])
nor_mat[,5] <- as.numeric(nor_mat[,5]) 
nor_mat
str(nor_mat)
set.seed(123)
#---------------train test split
ind <- sample(2,nrow(nor_mat),replace=T,prob=c(0.7,0.3))
ind
train <- nor_mat[ind==1,1:4]
train_label <- nor_mat[ind==1,5]
train_label
test  <- nor_mat[ind==2,1:4]
test_label <- nor_mat[ind==2,5]
test_label
trainLab <- to_categorical(train_label)
trainLab
testLab <- to_categorical(test_label)
testLab
#model

model <- keras_model_sequential()
model %>% 
  layer_dense(units=8,activation = "relu",input_shape = c(4)) %>%
  layer_dense(units=3,activation = "softmax")
summary(model)
model %>% compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics="accuracy")
#fit model
dim(train_label)

history <- model %>%
  fit(train,trainLab,epoch=200,batch_size=15)

plot(history)
#test data evaluation
model %>%
  evaluate(test,testLab)
###confusion matrix
prob <- model %>%
  predict_proba(test)
pred <- model %>%
  predict_classes(test)
table(predicted=pred,actual=test_label)
species
cbind(pred,test_label)

