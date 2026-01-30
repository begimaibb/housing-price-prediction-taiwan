library(knitr)
library(dplyr)
library(kableExtra)
library(webshot2)

# -------------------------------
# Combine metrics into a table
# -------------------------------
metrics_table <- data.frame(
  Model = c("MLR", "Random Forest", "XGBoost"),
  RMSE  = c(mlr_rmse, rf_rmse, xgb_rmse),
  R2    = c(mlr_r2, rf_r2, xgb_r2)
)

metrics_table <- metrics_table %>%
  mutate(across(where(is.numeric), ~ round(., 2)))


kbl_table <- kable(metrics_table, 
                   caption = "Comparison of RMSE and R² Across Models",
                   align = c("l", "r", "r")) %>%
  kable_styling(full_width = F, position = "center", bootstrap_options = c("striped", "hover"))


html_file <- tempfile(fileext = ".html")
save_kable(kbl_table, html_file)

# -------------------------------
# Convert table to PNG
# -------------------------------
png_file <- "results/model_metrics_table.png"

webshot(url = html_file,
        file = png_file,
        selector = "table",
        zoom = 2)
