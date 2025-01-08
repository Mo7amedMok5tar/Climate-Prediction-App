# Climate Prediction Project

This project builds a deep learning model to predict the weather conditions based on features like temperature, precipitation, and humidity. The model is trained using a dataset, and the app leverages this model to predict weather conditions for a selected date.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Preparation](#data-preparation)
   - [Loading the Dataset](#loading-the-dataset)
   - [Handling Missing Values and Outliers](#handling-missing-values-and-outliers)
   - [Feature Engineering](#feature-engineering)
   - [Handling Imbalanced Data using SMOTENC](#handling-imbalanced-data-using-smotenc)
3. [Data Visualization](#data-visualization)
   - [Feature Distribution](#feature-distribution)
   - [Pairplots to Understand Feature Relationships](#pairplots-to-understand-feature-relationships)
   - [Correlation Heatmap](#correlation-heatmap)
4. [Data Splitting](#data-splitting)
   - [Splitting Data into Training, Validation, and Test Sets](#splitting-data-into-training-validation-and-test-sets)
5. [Model Building](#model-building)
   - [Constructing the Deep Learning Model](#constructing-the-deep-learning-model)
   - [Model Architecture](#model-architecture)
6. [Model Training](#model-training)
   - [Compiling the Model](#compiling-the-model)
   - [Training the Model with Callbacks](#training-the-model-with-callbacks)
7. [Model Evaluation](#model-evaluation)
   - [Evaluating Performance using Confusion Matrix](#evaluating-performance-using-confusion-matrix)
   - [Accuracy and Loss Plots](#accuracy-and-loss-plots)
8. [User Interface](#user-interface)
   - [Streamlit-based Web App to Predict Weather](#streamlit-based-web-app-to-predict-weather)

## Project Overview

This project involves building and deploying a machine learning model to predict climate conditions (weather types). The model is trained using historical data, and it handles challenges such as imbalanced data using SMOTENC (Synthetic Minority Over-sampling Technique for Nominal and Continuous features). The goal is to predict weather categories like sun, fog, drizzle, rain, and snow.

## Data Preparation

### 1. Loading the Dataset
The dataset is loaded from a CSV file and initial checks for missing values or outliers are performed.

### 2. Handling Missing Values and Outliers
- Missing values are handled by appropriate imputation techniques.
- Outliers are detected and either capped or removed.

### 3. Feature Engineering
- New features such as `temp_precip_interaction`, `precip_wind_interaction`, and lag features (`temp_max_lag_1`, `precipitation_lag_1`) are created to provide more insight into weather patterns.

### 4. Handling Imbalanced Data
- SMOTENC is used to generate synthetic samples for the minority class in order to balance the dataset for model training.

## Data Visualization

### 1. Feature Distribution
- Histograms and other plots are used to visualize the distributions of features like precipitation, temperature, and wind speed.

### 2. Pairplots to Understand Feature Relationships
- Pairplots are used to explore the relationships between numerical features.

### 3. Correlation Heatmap
- A heatmap is plotted to visualize the correlation between features and understand their interactions.

## Data Splitting

The dataset is split into training, validation, and test sets with a stratified approach to maintain the class distribution of the target variable (`weather_encoded`).

## Model Building

### 1. Constructing the Deep Learning Model
The model is constructed using Keras. The architecture includes multiple layers with ReLU activations, batch normalization, dropout for regularization, and L2 regularization.

### 2. Model Architecture
The model consists of five Dense layers with progressively decreasing neurons (512, 256, 128, 64, and 32), followed by a final output layer with a number of neurons equal to the number of weather categories.

## Model Training

### 1. Compiling the Model
The model is compiled using the Adam optimizer and SparseCategoricalCrossentropy loss function, as it's a multi-class classification problem.

### 2. Training the Model with Callbacks
The model is trained using the SMOTE-sampled training data. Callbacks like early stopping, learning rate reduction, and model checkpointing are employed to prevent overfitting and ensure optimal performance.

## Model Evaluation

### 1. Accuracy and Loss Plots
Accuracy and loss curves for both the training and validation sets are plotted to visualize the model's learning process.

### 2. Confusion Matrix
A confusion matrix is generated for both the validation and test sets to evaluate the model’s classification performance. It shows how well the model is predicting each weather category.

## User Interface

### Streamlit-based Web App to Predict Weather

The web app allows users to input weather-related parameters and predict the weather for a selected date.

#### Inputs:
- Precipitation, max temperature, min temperature, wind speed, and historical weather conditions for the previous day are inputted through sliders and dropdown menus.

#### Predictions:
- The app predicts the weather condition based on the provided inputs. The result is shown with the predicted weather category (sun, fog, drizzle, rain, snow).

## Conclusion

This project demonstrates how deep learning can be applied to predict climate conditions based on weather-related features. The model was trained on historical data and is evaluated with multiple performance metrics. Additionally, a user-friendly Streamlit app was built to provide weather predictions based on user input.

**Note**: To run the app, ensure that the `scaler.pkl` and the trained model `Claimate_pred4.h5` are in the correct path.
