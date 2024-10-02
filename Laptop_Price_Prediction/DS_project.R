library(stats)
library(dplyr)
library(randomForest)
library(xgboost)
library(e1071)
library(ggplot2)
library(caret)
library(mlbench)
library(tree)

# Read the dataset
mydata <- read.csv("laptops.csv")
mydata<- subset(mydata, select = -11)

#for screen size
mydata$Screen.Size <- gsub("\"", "", mydata$Screen.Size)
mydata$Screen.Size <- as.numeric(mydata$Screen.Size)

#for RAM
mydata$RAM <- sub("GB", "", mydata$RAM)
mydata$RAM <- as.numeric(mydata$RAM)

#for Weight
mydata$Weight <- sub("kg", "", mydata$Weight)
mydata$Weight <- as.numeric(mydata$Weight)

# Splitting data into training and testing sets
set.seed(123)
index <- sample(2, nrow(mydata), replace = TRUE, prob = c(0.7, 0.3))
Training <- mydata[index == 1, ]
Testing <- mydata[index == 2, ]

# Random Forest Model
RFM <- randomForest(Price ~ ., data = Training, ntree = 100)
Price_Pred1 <- predict(RFM, Testing)

#XGboost
xgb_model <- xgboost(data = as.matrix(Training[, "Price"]),label = Training$Price,nrounds = 100,objective = "reg:squarederror", eta = 0.1, max_depth = 6)
Price_Pred2 <- predict(xgb_model, as.matrix(Testing[, "Price"]))


#decission Tree
DTM <- tree(Price ~ ., data = Training)

# Predict using the decision tree model
Price_Pred3 <- predict(DTM, Testing)

Testing$Price_Pred1 = Price_Pred1
Testing$Price_Pred2 = Price_Pred2
Testing$Price_Pred3 = Price_Pred3


mse1 <- mean((Price_Pred1 - Testing$Price)^2)
rmse1 <- sqrt(mse1)

mse2 <- mean((Testing[, "Price"] - Price_Pred2)^2)
rmse2 <- sqrt(mse2)

mse3 <- mean((Price_Pred3 - Testing$Price)^2)
rmse3 <- sqrt(mse3)

#SVR
#SVR_model <- svm(Price ~ .,data =  Training,kernel = 'radial')
#Price_Pred3 <- predict(SVR_model,newdata =  Testing)



cat("Mean Squared Error (MSE) for RFM :", mse1, "\n")
cat("Root Mean Squared Error (RMSE) for RFM:", rmse1, "\n\n")
cat("Mean Squared Error (MSE) for XGboost :", mse2, "\n")
cat("Root Mean Squared Error (RMSE) for XGboost:", rmse2, "\n\n")
cat("Mean Squared Error (MSE) for Decision Tree:", mse3, "\n")
cat("Root Mean Squared Error (RMSE) for Decision Tree:", rmse3, "\n\n")



# View the Testing dataset with prediction columns
View(Testing)

control <- rfeControl(functions = rfFuncs, method = "cv", number = 10)
# You can change the 'number' parameter to specify the number of cross-validations.

# Create a data frame without the 'Price' column as the target variable
TrainingWithoutPrice <- Training[, -which(names(Training) == "Price")]

# Perform RFE for the Random Forest model
rfe_rf <- rfe(TrainingWithoutPrice, Training$Price, sizes = c(1:ncol(TrainingWithoutPrice)),
              rfeControl = control)
# 'sizes' specifies the number of features to select. In this example, it selects from 1 to the number of columns.

# Output the results
print(rfe_rf)

# Identify the selected features for Random Forest
selected_features_rf <- rfe_rf$optVariables
print(selected_features_rf)

# Perform RFE for the XGBoost model
rfe_xgb <- rfe(TrainingWithoutPrice, Training$Price, sizes = c(1:ncol(TrainingWithoutPrice)),
               rfeControl = control, method = "xgbLinear")
# 'method' specifies the method to use. In this case, "xgbLinear" for XGBoost linear regression.

# Output the results
print(rfe_xgb)

# Identify the selected features for XGBoost
selected_features_xgb <- rfe_xgb$optVariables
print(selected_features_xgb)

r_sq.lm= lm(Testing$Price~Price_Pred1, data = Testing)
r_sq.lm_sum_1 <- summary(r_sq.lm)$r.squared
cat("\n r_sq_1 : ",r_sq.lm_sum_1)

r_sq.lm= lm(Testing$Price~Price_Pred2, data = Testing)
r_sq.lm_sum_2 <- summary(r_sq.lm)$r.squared
cat("\n r_sq_2 : ",r_sq.lm_sum_2)

r_sq.lm= lm(Testing$Price~Price_Pred3, data = Testing)
r_sq.lm_sum_3 <- summary(r_sq.lm)$r.squared
cat("\n r_sq_3 : ",r_sq.lm_sum_3)
cat("\n\n\n")

# Create a data frame with the actual and predicted values for the first 50 observations
plot_data <- data.frame(
  Observed = Testing$Price[1:50],
  Predicted_RFM = Price_Pred1[1:50],
  Predicted_XGBoost = Price_Pred2[1:50],
  Predicted_DecisionTree = Price_Pred3[1:50]
)

# Create a line graph with adjusted transparency
library(ggplot2)

ggplot(plot_data, aes(x = 1:length(Observed))) +
  geom_line(aes(y = Observed, color = "Actual"), size = 1) +
  geom_line(aes(y = Predicted_RFM, color = "Predicted 1"), size = 1, alpha = 0.5) +
  geom_line(aes(y = Predicted_XGBoost, color = "Predicted 2"), size = 1, alpha = 0.5) +
  geom_line(aes(y = Predicted_DecisionTree, color = "Predicted 3"), size = 1, alpha = 0.5) +
  labs(title = "Actual vs. Predicted Values (First 50 Observations)", x = "Index", y = "Price") +
  scale_color_manual(
    values = c("Actual" = "black", "Predicted 1" = "red", "Predicted 2" = "green", "Predicted 3" = "purple")
  )

## RMSE bar graph
# Create a data frame with the RMSE values
rmse_data <- data.frame(
  Model = c("Model 1", "Model 2", "Model 3"),
  RMSE = c(rmse1, rmse2, rmse3)
)

# Create a bar graph for RMSE values
ggplot(rmse_data, aes(x = Model, y = RMSE)) +
  geom_bar(stat = "identity", fill = "blue") +
  labs(title = "RMSE Comparison", y = "RMSE Value") +
  theme_minimal()


## MSE bar graph
# Create a data frame with the MSE values
mse_data <- data.frame(
  Model = c("Model 1", "Model 2", "Model 3"),
  MSE = c(mse1, mse2, mse3)
)

# Create a bar graph for MSE values
ggplot(mse_data, aes(x = Model, y = MSE)) +
  geom_bar(stat = "identity", fill = "red") +
  labs(title = "MSE Comparison", y = "MSE Value") +
  theme_minimal()


## R square bar graph  
# Create a data frame with the R-squared values
rsquared_data <- data.frame(
  Model = c("Model 1", "Model 2", "Model 3"),
  R_squared = c(r_sq.lm_sum_1, r_sq.lm_sum_2, r_sq.lm_sum_3)
)

# Create a bar graph for R-squared values
ggplot(rsquared_data, aes(x = Model, y = R_squared)) +
  geom_bar(stat = "identity", fill = "green") +
  labs(title = "R-squared Comparison", y = "R-squared Value") +
  theme_minimal()
