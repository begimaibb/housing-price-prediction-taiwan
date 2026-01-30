# ===============================
# 03_model_training_and_evaluation_RF.R
# Random Forest Regression with Visualizations
# ===============================

library(dplyr)
library(randomForest)
library(caret)
library(Metrics)
library(ggplot2)

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

rf_rmse <- rmse(actual_prices, predicted_prices)
rf_r2   <- 1 - sum((actual_prices - predicted_prices)^2)/sum((actual_prices - mean(actual_prices))^2)



# -------------------------------
# Visualizations
# -------------------------------

# Predicted vs Actual (log scale)
png("results/figures/rf_pred_vs_actual_log.png", width = 900, height = 600)
ggplot(data = data.frame(Actual = Y_test, Predicted = predictions_rf), aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.5, color = "blue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "Random Forest: Predicted vs Actual (log scale)",
       x = "Actual sqmPriceL",
       y = "Predicted sqmPriceL") +
  theme_minimal()
dev.off()

# Predicted vs Actual (original scale)
png("results/figures/rf_pred_vs_actual_exp.png", width = 900, height = 600)
ggplot(data = data.frame(Actual = actual_prices, Predicted = predicted_prices), aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.5, color = "darkgreen") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "Random Forest: Predicted vs Actual (original scale)",
       x = "Actual sqmPrice (in NTD)",
       y = "Predicted sqmPrice (in NTD)") +
  theme_minimal()
dev.off()

# Variable Importance
png("results/figures/rf_variable_importance.png", width = 900, height = 600)
varImpPlot(rf_tuned$finalModel, main = "Random Forest Variable Importance")
dev.off()
