# ===============================
# 03_model_training_and_evaluation_RF.R
# Random Forest Regression
# ===============================

library(dplyr)
library(randomForest)
library(caret)
library(Metrics)

set.seed(1234)

housing <- read_csv("data/PROCESSED_new_taipei_housing_transactions_2018_2023.csv")


# -------------------------------
# Feature selection & cleaning
# -------------------------------
housing_X <- housing %>%
  select(
    sqmPriceL,
    buildingAreaL,
    age,
    totalFloors,
    district,
    buildingType,
    parkingAreaL,
    balconyAreaL,
    landUseZoning,
    parkingCategory
  ) %>%
  filter(if_all(everything(), ~ !is.na(.) & !is.nan(.) & !is.infinite(.))) %>%
  droplevels()

# -------------------------------
# Train / Test split (80 / 20)
# -------------------------------
smp_size <- floor(0.8 * nrow(housing_X))
train_ind <- sample(seq_len(nrow(housing_X)), size = smp_size)

train <- housing_X[train_ind, ]
test  <- housing_X[-train_ind, ]

Y_test <- test$sqmPriceL
test$sqmPriceL <- NULL

# -------------------------------
# Hyperparameter tuning
# -------------------------------
train_control <- trainControl(
  method = "cv",
  number = 3
)

mtry_grid <- expand.grid(
  mtry = c(2, 4, 6)
)

rf_tuned <- train(
  sqmPriceL ~ .,
  data = train,
  method = "rf",
  trControl = train_control,
  tuneGrid = mtry_grid,
  ntree = 300,
  importance = TRUE
)

print(rf_tuned)

# -------------------------------
# Predictions
# -------------------------------
predictions_rf <- predict(rf_tuned, test)

# -------------------------------
# Evaluation (log scale)
# -------------------------------
r2_log <- 1 - sum((Y_test - predictions_rf)^2) /
  sum((Y_test - mean(Y_test))^2)

cat("\nRandom Forest – Log scale:\n")
cat("MAE :", mae(Y_test, predictions_rf), "\n")
cat("MSE :", mse(Y_test, predictions_rf), "\n")
cat("RMSE:", rmse(Y_test, predictions_rf), "\n")
cat("R²  :", r2_log, "\n")

# -------------------------------
# Evaluation (original scale)
# -------------------------------
actual_prices <- exp(Y_test)
predicted_prices <- exp(predictions_rf)

r2_exp <- 1 - sum((actual_prices - predicted_prices)^2) /
  sum((actual_prices - mean(actual_prices))^2)

cat("\nRandom Forest – Original scale:\n")
cat("MAE :", mae(actual_prices, predicted_prices), "\n")
cat("MSE :", mse(actual_prices, predicted_prices), "\n")
cat("RMSE:", rmse(actual_prices, predicted_prices), "\n")
cat("R²  :", r2_exp, "\n")

# -------------------------------
# Variable Importance
# -------------------------------
varImpPlot(rf_tuned$finalModel)
