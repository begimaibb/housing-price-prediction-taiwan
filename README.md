# Housing Price Prediction in Taiwan

**Predicting Housing Prices in Taiwan: A Comparative Evaluation of Machine Learning Models**

This repository contains my MBA thesis project, which focuses on predicting housing prices in Taiwan using multiple machine learning models. The project evaluates and compares traditional and advanced ML techniques to identify the most effective approach for real estate price prediction.

---

## Project Overview

Accurate housing price prediction is essential for buyers, sellers, and policymakers. This project analyzes Taiwanese housing data and applies various machine learning models to predict property prices, comparing their performance based on standard regression metrics.

The study emphasizes:
- Data preprocessing and feature engineering
- Model comparison and evaluation
- Practical insights for real-world real estate applications

---

## Objectives

- Explore key factors influencing housing prices in Taiwan
- Build and evaluate multiple machine learning models
- Compare model performance using statistical metrics
- Identify the best-performing model for price prediction

---

## Dataset

- **Source**: Taiwan Ministry of Digital Affairs open data platform
- **Target Variable**:  Sale price / price per square meter
- **Features: ** 32 attributes including:
  - Location-related variables
  - Property age
  - Number of rooms
  - Distance to amenities
  - Transaction-related attributes


---

## Models Implemented

The following models were implemented and compared:

- Linear Multiple Regression
- Random Forest Regressor
- Gradient Boosting Models

---

## Evaluation Metrics

Model performance was evaluated using:

- R² Score
- Root Mean Squared Error (RMSE)

---

## Technologies & Tools

- **Programming Languages**: R
- **Libraries (R)**:
  - **dplyr** – data manipulation and transformation
  - **ggplot2** – data visualization
  - **Metrics** – model evaluation metrics
  - **corrplot** – correlation analysis and visualization
  - **readr** – data import
  - **lubridate** – date and time feature handling
  - **caret** – machine learning modeling and validation
- **Environment**:  RStudio

---

## Repository Structure

```text
housing-price-prediction-taiwan/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── 01_data_exploration.R
│   ├── 02_preprocessing.R
│   ├── 03_model_training_and_evaluation_MLR.R
│   ├── 03_model_training_and_evaluation_RF.R
│   ├── 03_model_training_and_evaluation_XGB.R
│   └── 04_display_metrics.R
├── results/
│   ├── figures/
│   └── model_metrics_table.png 
├── README.md
├── LICENSE
└── .gitignore

