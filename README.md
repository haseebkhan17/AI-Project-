# 🦴 Spine Condition Classification using Linear Regression

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-green)
![GitHub issues](https://img.shields.io/badge/GitHub-Issues-lightgrey)

## Project Overview

Spinal disorders are a significant health concern worldwide, often leading to pain, reduced mobility, and impaired quality of life. Early diagnosis is crucial for effective treatment.  

This project leverages **machine learning** to classify spinal conditions using a **biomechanical dataset of the human spine**. A **Linear Regression** model is trained on spinal features and adapted for **binary classification** by applying a threshold on predicted values.  

The goal is to **distinguish between normal and abnormal spinal conditions**, evaluate the model’s performance using metrics, and visualize the results for clear interpretation. This project serves as an academic example of **machine learning workflow in healthcare data analysis**.

## Dataset Description

- **Dataset Name:** Spine Dataset  
- **Source:** [Kaggle](https://www.kaggle.com)  
- **Number of Rows / Columns:** 310 rows, 6 features + 1 target  

### Features

| Feature | Description |
|---------|-------------|
| Pelvic Incidence | Angular measurement of pelvis orientation |
| Pelvic Tilt | Degree of tilt of the pelvis |
| Lumbar Lordosis Angle | Curvature of the lower spine |
| Sacral Slope | Inclination of the sacrum |
| Pelvic Radius | Distance measurement related to pelvis geometry |
| Degree of Spondylolisthesis | Severity of vertebra slippage |

### Target Variable

- **Class_att**: 0 = Normal, 1 = Abnormal  

All features are numeric and suitable for regression-based modeling.  

## Data Preprocessing

1. Checked and verified that there are no missing values.  
2. Converted `Class_att` target variable into numeric format.  
3. Split dataset into training (75%) and testing (25%) sets.  
4. Selected all biomechanical features for model training.  
5. Optional: Scaling can be applied for numerical stability.

## Model Training

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict values
y_pred = lr.predict(X_test)

# Convert predictions to binary
y_pred_binary = [1 if val >= 0.5 else 0 for val in y_pred]

Evaluation Metrics

Accuracy: Proportion of correct predictions

Confusion Matrix: True positives, true negatives, false positives, false negatives

Scatter Plots: Compare actual vs predicted values

from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Accuracy
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_binary)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

Visualizations

Feature correlation heatmap

Confusion matrix heatmap

Actual vs predicted scatter plot

Visualizations help interpret results and communicate model insights clearly.

Repository Contents

Dataset_spine.csv – Dataset file

Spine_Linear_Regression.ipynb – Notebook with code, metrics, and visualizations

README.md – Project documentation and instructions

Instructions to Run

Open Spine_Linear_Regression.ipynb in Google Colab or Jupyter Notebook.

Upload Dataset_spine.csv.

Run all cells to:

Load and preprocess data

Train the Linear Regression model

Evaluate metrics and view graphs

Adjust the classification threshold to observe its effect on accuracy and confusion matrix outcomes.

Potential Improvements

Use Logistic Regression, Decision Trees, or Random Forests for improved classification

Apply feature scaling or normalization for numerical stability

Use cross-validation to improve model reliability

Explore feature importance to identify the most influential measurements

Experiment with ensemble methods for higher accuracy

Conclusion

This project demonstrates how Linear Regression can be adapted for classification tasks in healthcare data analysis. Through preprocessing, training, evaluation, and visualization, it provides a strong foundation for spinal condition prediction and serves as an academic reference for machine learning workflows in medical datasets.

Author

Name: Muhammad Haseeb Khan

University: Capital University of Science & Technology (CUST)

GitHub: haseebkhan17
