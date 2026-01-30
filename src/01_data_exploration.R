# ===============================
# 01_data_exploration.R
# Exploratory Data Analysis
# ===============================

library(dplyr)
library(ggplot2)
library(readr)
library(corrplot)

set.seed(1234)

# Load raw data
housing <- read_csv("data/RAW_new_taipei_housing_transactions_2018_2023.csv") 

# Structure and summary
str(housing)
summary(housing)

# Missing values overview
sapply(housing, function(x) sum(is.na(x)))

# Price distributions
ggplot(housing, aes(x = rps21)) +
  geom_histogram(binwidth = 10000) +
  labs(title = "Total Price Distribution")

ggplot(housing, aes(x = rps22)) +
  geom_histogram(binwidth = 10000) +
  labs(title = "Price per sqm Distribution")

