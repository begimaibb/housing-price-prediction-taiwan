# ===============================
# 03_model_training_and_evaluation_XGB.R
# XGBoost Regression with Visualizations
# ===============================

library(dplyr)
library(caret)
library(xgboost)
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
# One-hot encoding
# -------------------------------
dummy_vars <- dummyVars(sqmPriceL ~ ., data = housing_X)
housing_matrix <- predict(dummy_vars, housing_X) %>% as.matrix()

# Target variable
Y <- housing_X$sqmPriceL

# -------------------------------
# Train / Test split (80 / 20)
# -------------------------------
smp_size <- floor(0.8 * nrow(housing_matrix))
train_ind <- sample(seq_len(nrow(housing_matrix)), size = smp_size)

X_train <- housing_matrix[train_ind, ]
X_test  <- housing_matrix[-train_ind, ]

Y_train <- Y[train_ind]
Y_test  <- Y[-train_ind]

# -------------------------------
# DMatrix conversion
# -------------------------------
dtrain <- xgb.DMatrix(data = X_train, label = Y_train)
dtest  <- xgb.DMatrix(data = X_test)

# -------------------------------
# Model parameters
# -------------------------------
params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  eta = 0.05,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# -------------------------------
# Train model
# -------------------------------
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 300,
  verbose = 0
)

# -------------------------------
# Predictions
# -------------------------------
predictions_xgb <- predict(xgb_model, dtest)

# -------------------------------
# Evaluation (log scale)
# -------------------------------
r2_log <- 1 - sum((Y_test - predictions_xgb)^2) /
  sum((Y_test - mean(Y_test))^2)

cat("\nXGBoost – Log scale:\n")
cat("MAE :", mae(Y_test, predictions_xgb), "\n")
cat("MSE :", mse(Y_test, predictions_xgb), "\n")
cat("RMSE:", rmse(Y_test, predictions_xgb), "\n")
cat("R²  :", r2_log, "\n")

# -------------------------------
# Evaluation (original scale)
# -------------------------------
actual_prices <- exp(Y_test)
predicted_prices <- exp(predictions_xgb)

r2_exp <- 1 - sum((actual_prices - predicted_prices)^2) /
  sum((actual_prices - mean(actual_prices))^2)

cat("\nXGBoost – Original scale:\n")
cat("MAE :", mae(actual_prices, predicted_prices), "\n")
cat("MSE :", mse(actual_prices, predicted_prices), "\n")
cat("RMSE:", rmse(actual_prices, predicted_prices), "\n")
cat("R²  :", r2_exp, "\n")


xgb_rmse <- rmse(actual_prices, predicted_prices)
xgb_r2   <- 1 - sum((actual_prices - predicted_prices)^2)/sum((actual_prices - mean(actual_prices))^2)


# -------------------------------
# Visualizations
# -------------------------------

# Predicted vs Actual (log scale)
png("results/figures/xgb_pred_vs_actual_log.png", width = 900, height = 600)
ggplot(data = data.frame(Actual = Y_test, Predicted = predictions_xgb), aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.5, color = "blue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "XGBoost: Predicted vs Actual (log scale)",
       x = "Actual sqmPriceL",
       y = "Predicted sqmPriceL") +
  theme_minimal()
dev.off()

# Predicted vs Actual (original scale)
png("results/figures/xgb_pred_vs_actual_exp.png", width = 900, height = 600)
ggplot(data = data.frame(Actual = actual_prices, Predicted = predicted_prices), aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.5, color = "darkgreen") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "XGBoost: Predicted vs Actual (original scale)",
       x = "Actual sqmPrice (in NTD)",
       y = "Predicted sqmPrice (in NTD)") +
  theme_minimal()
dev.off()

# Feature Importance
xgb_imp <- xgb.importance(model = xgb_model)

png("results/figures/xgb_feature_importance.png", width = 900, height = 600)
xgb.plot.importance(xgb_imp, top_n = 15, main = "XGBoost Feature Importance")
dev.off()
