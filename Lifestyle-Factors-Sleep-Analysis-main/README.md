**Project Overview**

This project analyzes how different lifestyle factors influence sleep quality and sleep duration. Using the Sleep Health and Lifestyle Dataset from Kaggle, the project performs data analysis, visualization, and machine learning prediction.

The goal is to understand relationships between lifestyle habits and sleep health and build a machine learning model that predicts sleep quality based on lifestyle inputs.

**Objectives**:
Analyze the relationship between lifestyle factors and sleep patterns.
Identify key factors affecting sleep quality.
Build a machine learning model to predict sleep quality.
Visualize patterns and correlations in the dataset.
Dataset

**Dataset used:**
Sleep Health and Lifestyle Dataset

It contains features such as:
Age
Gender
Occupation
Sleep Duration
Physical Activity Level
Stress Level
BMI Category
Heart Rate
Daily Steps
Sleep Disorder
Sleep Quality (Target Variable)

Dataset source:
https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset

Technologies Used
Python
Pandas (data preprocessing)
NumPy
Matplotlib
Seaborn
Scikit-learn (Machine Learning)
Jupyter Notebook / VS Code

**Machine Learning Model**:
A Random Forest Regressor model is used to predict Sleep Quality based on lifestyle factors.

Steps followed:
Data preprocessing
Handling categorical variables
Feature selection
Train-test split
Model training
Prediction of sleep quality
Key Visualizations

The project includes several visual analyses:
Sleep Duration vs Sleep Quality
Stress Level vs Sleep Duration
Physical Activity vs Sleep Duration
BMI Category vs Sleep Quality
Sleep Disorder vs Sleep Duration
These visualizations help understand how lifestyle habits affect sleep.

Model Prediction-The trained machine learning model can predict sleep quality when given lifestyle inputs such as:
Sleep duration
Stress level
Physical activity
BMI category
Daily steps
Heart rate

Example output:
Predicted Sleep Quality Score: 7.8
and provide the recommendations also.
**How to Run the Project**:
Step 1

Clone the repository

git clone https://github.com/your-username/Lifestyle-Factors-Sleep-Analysis.git
Step 2

Install required libraries

pip install pandas numpy matplotlib seaborn scikit-learn
Step 3

Run the project

python sleep_quality.py

or open the notebook and run all cells.

**Key Insights**:
Higher stress levels are linked to lower sleep duration.
Individuals with higher physical activity tend to sleep longer.
BMI category shows a relationship with sleep quality.

Sleep disorders negatively impact sleep duration.

Lifestyle factors significantly influence sleep health.
