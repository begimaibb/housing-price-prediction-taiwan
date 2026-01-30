# ===============================
# 03_model_training_and_evaluation_MLR.R
# Model Training & Evaluation with Recommendations
# ===============================

library(dplyr)
library(readr)
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


mlr_rmse <- rmse(actual_prices, predicted_prices)
mlr_r2   <- 1 - sum((actual_prices - predicted_prices)^2)/sum((actual_prices - mean(actual_prices))^2)


# ===============================
# Visualizations
# ===============================

# Predicted vs Actual (log scale)
png("results/figures/mlr_pred_vs_actual_log.png", width = 900, height = 600)
ggplot(data.frame(Actual = Y_test, Predicted = predictions), aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue", alpha = 0.4) +
  geom_abline(slope = 1, intercept = 0, color = "red", size = 1.2) +
  labs(
    title = "MLR: Predicted vs Actual (log scale)",
    x = "Actual sqmPriceL",
    y = "Predicted sqmPriceL"
  ) +
  theme_minimal()
dev.off()

# Predicted vs Actual (original scale)
png("results/figures/mlr_pred_vs_actual_exp.png", width = 900, height = 600)
ggplot(data.frame(Actual = actual_prices, Predicted = predicted_prices), aes(x = Actual, y = Predicted)) +
  geom_point(color = "darkgreen", alpha = 0.4) +
  geom_abline(slope = 1, intercept = 0, color = "red", size = 1.2) +
  labs(
    title = "MLR: Predicted vs Actual (original scale)",
    x = "Actual sqmPrice(in NTD)",
    y = "Predicted sqmPrice(in NTD)"
  ) +
  theme_minimal()
dev.off()

# Residuals plot
residuals <- Y_test - predictions
png("results/figures/mlr_residuals.png", width = 900, height = 600)
ggplot(data.frame(Fitted = predictions, Residuals = residuals), aes(x = Fitted, y = Residuals)) +
  geom_point(alpha = 0.4) +
  geom_hline(yintercept = 0, color = "red") +
  labs(
    title = "MLR: Residuals vs Fitted",
    x = "Fitted values",
    y = "Residuals"
  ) +
  theme_minimal()
dev.off()
