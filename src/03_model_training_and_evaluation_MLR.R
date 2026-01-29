# ===============================
# 03_model_training_and_evaluation_MLR.R
# Model Training & Evaluation
# ===============================

library(dplyr)
library(readr)
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
# Train / Test Split (80 / 20)
# -------------------------------
smp_size <- floor(0.8 * nrow(housing_X))
train_ind <- sample(seq_len(nrow(housing_X)), size = smp_size)

train <- housing_X[train_ind, ]
test  <- housing_X[-train_ind, ]

Y_test <- test$sqmPriceL
test$sqmPriceL <- NULL

# -------------------------------
# Linear Regression
# -------------------------------
linearModel <- lm(sqmPriceL ~ ., data = train)

summary(linearModel)

# -------------------------------
# Predictions
# -------------------------------
predictions <- predict(linearModel, test)

# -------------------------------
# Evaluation (log scale)
# -------------------------------
log_mse  <- mean((Y_test - predictions)^2)
log_rmse <- rmse(Y_test, predictions)
log_mae  <- mean(abs(Y_test - predictions))

cat("\nLog-scale performance:\n")
cat("Log-MSE :", log_mse, "\n")
cat("Log-RMSE:", log_rmse, "\n")
cat("Log-MAE :", log_mae, "\n")

# -------------------------------
# Evaluation (original scale)
# -------------------------------
actual_prices    <- exp(Y_test)
predicted_prices <- exp(predictions)

mse  <- mean((actual_prices - predicted_prices)^2)
rmse_val <- rmse(actual_prices, predicted_prices)
mae_val  <- mean(abs(actual_prices - predicted_prices))

cat("\nOriginal scale performance (sqm price):\n")
cat("MSE :", mse, "\n")
cat("RMSE:", rmse_val, "\n")
cat("MAE :", mae_val, "\n")

# -------------------------------
# Preview predictions
# -------------------------------
head(
  data.frame(
    Actual = actual_prices,
    Predicted = predicted_prices
  ),
  10
)
