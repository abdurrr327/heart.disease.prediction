# Heart Disease Prediction Project

This repository contains a Python project, designed to be run in Google Colab, for predicting the presence of heart disease using a classic machine learning approach.

## 1. Task Objective
The primary objective of this project is to build and evaluate a binary classification model that can predict whether a person is at risk of heart disease based on their medical and demographic data. The project covers the entire machine learning workflow, from data cleaning and exploration to model training, evaluation, and interpretation.

## 2. Dataset Used
The project utilizes the **Heart Disease UCI Dataset**, specifically from the provided `HeartDiseaseTrain-Test.csv` file.

*   **Source:** A modified version of the popular dataset from the UCI Machine Learning Repository, available on platforms like Kaggle.
*   **Instances:** 1025
*   **Features:** 14 attributes, including:
    *   `age`: Age of the patient
    *   `sex`: Sex of the patient (Male/Female)
    *   `chest_pain_type`: The type of chest pain experienced
    *   `resting_blood_pressure`: Resting blood pressure (in mm Hg)
    *   `cholesterol`: Serum cholesterol (in mg/dl)
    *   `fasting_blood_sugar`: Indicator if fasting blood sugar > 120 mg/dl
    *   `Max_heart_rate`: Maximum heart rate achieved
    *   `target`: The diagnosis of heart disease (0 = No, 1 = Yes)

## 3. Models and Methodology

The project follows a structured machine learning pipeline:

1.  **Data Cleaning:** The dataset was inspected for missing values and inconsistencies. A column name typo (`cholestoral` -> `cholesterol`) was corrected, and a minor data anomaly in the `thalassemia` column was handled by replacing it with the mode.

2.  **Exploratory Data Analysis (EDA):** Visualizations like count plots, histograms, and heatmaps were used to understand trends, distributions, and correlations between features and the target variable.

3.  **Data Preprocessing:**
    *   Categorical features with text values were converted into a numerical format.
    *   Binary and ordinal features (`sex`, `fasting_blood_sugar`, etc.) were mapped to integers.
    *   Nominal features (`chest_pain_type`, `slope`, etc.) were one-hot encoded to create separate binary columns for each category.
    *   Features were scaled using `StandardScaler` to prepare the data for Logistic Regression.

4.  **Models Applied:**
    Two common and interpretable classification models were trained:
    *   **Logistic Regression:** A linear model that is effective for binary classification and provides easily interpretable coefficients.
    *   **Decision Tree Classifier:** A non-linear, tree-based model that makes predictions by following a series of feature-based rules. Its structure makes it highly interpretable.

## 4. Key Results and Findings

### Model Performance
Both models performed well and demonstrated strong predictive power, far exceeding a random guess.

*   **Logistic Regression:**
    *   **Accuracy:** ~85.4%
    *   **ROC-AUC Score:** ~0.90
*   **Decision Tree:**
    *   **Accuracy:** ~86.8%
    *   **ROC-AUC Score:** ~0.89

The high AUC scores for both models indicate they are excellent at distinguishing between patients with and without heart disease.

### Important Features Affecting Prediction
Feature importance analysis was conducted for both models to identify the most influential factors. The findings were consistent across both models, highlighting the following features as key predictors:

1.  **`vessels_colored_by_flourosopy`**: The number of major vessels visible on a flouroscopy scan was a top predictor.
2.  **`chest_pain_type`**: The specific type of chest pain (or lack thereof) was highly indicative of heart disease risk.
3.  **`thalassemia`**: The type of thalassemia blood disorder (especially "Reversable Defect") was a significant factor.
4.  **`Max_heart_rate`**: The maximum heart rate achieved by the patient during exercise.
5.  **`oldpeak`**: The ST depression induced by exercise.
6.  **`sex`**: Gender was consistently identified as an important predictive feature.

These findings align with medical knowledge, confirming that the models learned relevant patterns from the data.
