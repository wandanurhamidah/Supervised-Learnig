##### CLASSIFICATION ON IRIS DATASET

### Importing Data
iris <- read.csv("~/DSC/Pertemuan 6/iris.csv", row.names=1)
View(iris)
features <- iris[,1:4] #define features
features

### EDA
View(iris) # to view the iris dataset
str(iris) # to view the structure of the iris dataset
unique(iris[c("Species")]) # to check the class on the species column
table(iris$Species) # to count the number of observationsin each class only

### Preprocessing Data
# Detecting Missing Value
summary(iris)

# Detecting Outlier
boxplot(features, main = "Detecting outliers on Each Variable")

# Detecting Multivariate Outlier
library(outliers)
z <- abs(scores(features, type = 'z'))
outlier <- subset(z, SepalLengthCm > 3 | SepalWidthCm > 3 | PetalLengthCm > 3 | PetalWidthCm > 3)
outlier
outlier_df <- data.frame("Notes" = c("Not Outlier", "Outlier"), "Number of observations" = c(dim(iris)[1]-dim(outlier)[1], dim(outlier)[1]))
outlier_df
barplot(outlier_df$Number.of.observations, names.arg = outlier_df$Notes, main = "Detecting Multivariate Outliers", ylab = "Number of observations")

# Normalization using L2 Norm
l2_norm_func <- function(x) sqrt(sum(x^2)) # to build the l2 norm function
l2_norm_func
l2_norm <- apply(features,1, l2_norm_func)
l2_norm
features_normalized <- features/l2_norm
features_normalized

# Detecting Outlier After Normalization
boxplot(features_normalized, main = "Detecting Outliers on Each Variable After Normalization")

# Detecting Multivariate Outlier After Normalization
z_norm <- abs(scores(features_normalized, type = 'z'))
outlier_norm <- subset(z_norm, SepalLengthCm > 3 | SepalWidthCm > 3 | PetalLengthCm > 3 | PetalWidthCm > 3)
outlier_norm_df <- data.frame("Notes" = c("Not Outlier", "Outlier"), "Number of observations" = c(dim(iris)[1]-dim(outlier)[1], dim(outlier)[1]))
outlier_norm_df
barplot(outlier_norm_df$Number.of.observations, names.arg = outlier_df$Notes, main = "Detecting Multivariate Outliers on Each Variable After Normalization", ylab = "Number of observations")

### Classification using K-Nearest Neighbor
# Combining Features Normalized and Label
iris_normalized <- iris
iris_normalized[1:4] <- features_normalized

# Splitting Data into Training and Testing
set.seed(101)
library(caTools)
split <- sample.split(iris_normalized, SplitRatio = 0.8)
train_set <- subset(iris_normalized, split == TRUE)
test_set <- subset(iris_normalized, split == FALSE)
dim(train_set)
dim(test_set)
x_train <- train_set[,1:4]
x_test <- test_set[,1:4]
y_train <- train_set[,5]
y_test <- test_set[,5]

# Classification using K-Nearest NNeighbor
set.seed(0)
library(class)
k = 3
knn.pred = knn(x_train, x_test, y_train, k = k)

#Confusion Matrix and Model Performance
library(caret)
library(e1071)
cm <- table(knn.pred, y_test)
confusionMatrix(cm)
