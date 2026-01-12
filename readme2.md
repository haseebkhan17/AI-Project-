# 🦴 Spine Condition Classification using Linear Regression

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-green) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🌟 Project Overview

Spinal disorders are a significant health concern worldwide, often leading to pain, reduced mobility, and impaired quality of life. Early diagnosis and monitoring are crucial for effective treatment. This project leverages **machine learning techniques** to classify spinal conditions using a **biomechanical dataset of the human spine**.

A **Linear Regression** model is trained on biomechanical features and adapted for **binary classification** by applying a threshold on predicted values. The main goal is to **distinguish between normal and abnormal spinal conditions** and evaluate the model’s performance using metrics and visualizations. This serves as a strong academic example for applying regression techniques to medical data.

---

## 📊 Dataset Description

**Dataset Name:** Spine Dataset  
**Source:** [Kaggle](https://www.kaggle.com)  
**Number of Rows / Columns:** 310 rows, 6 features + 1 target  

### Features:
- **Pelvic Incidence**: Angular measurement of pelvis orientation  
- **Pelvic Tilt**: Degree of tilt of the pelvis  
- **Lumbar Lordosis Angle**: Curvature of the lower spine  
- **Sacral Slope**: Inclination of the sacrum  
- **Pelvic Radius**: Distance measurement related to pelvis geometry  
- **Degree of Spondylolisthesis**: Severity of vertebra slippage  

### Target Variable:
- **Class_att** (0 = Normal, 1 = Abnormal)

**Observation:**  
All features are numeric, continuous values, making them suitable for **regression-based modeling**. These measurements are clinically significant and commonly used in spinal health assessment.

---

## 🧹 Data Preprocessing

1. **Missing Value Check**: Verified that no null or missing values existed.  
2. **Label Encoding**: Converted the `Class_att` target variable into numeric format.  
3. **Data Splitting**: Divided the dataset into **training (75%)** and **testing (25%)** sets.  
4. **Feature Selection**: All biomechanical features were selected for model training.  
5. **Normalization/Scaling (Optional)**: While not strictly necessary for Linear Regression, scaling can improve numerical stability for models in future iterations.  

---

## 🎯 Features and Target Definition

- **Features (X):** All spinal measurements (`Pelvic Incidence`, `Pelvic Tilt`, `Lumbar Lordosis Angle`, `Sacral Slope`, `Pelvic Radius`, `Degree of Spondylolisthesis`)  
- **Target (y):** `Class_att` (Binary: 0 = Normal, 1 = Abnormal)

---

## 🛠 Model Training

- **Algorithm Used:** Linear Regression  
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`  

```python
# Import libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv("Dataset_spine.csv")

# Features and target
X = data[['Pelvic Incidence','Pelvic Tilt','Lumbar Lordosis Angle','Sacral Slope','Pelvic Radius','Degree of Spondylolisthesis']]
y = data['Class_att']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize and train Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict on test set
y_pred = lr.predict(X_test)

# Convert continuous predictions to binary classes
y_pred_binary = [1 if val >= 0.5 else 0 for val in y_pred]
📏 Evaluation Metrics
The model was evaluated using standard metrics for classification:

python
Copy code
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
Accuracy: Proportion of correct predictions

Confusion Matrix: True positives, true negatives, false positives, false negatives

Scatter Plots: Compare actual vs predicted values

📈 Visualizations
Feature Correlation Heatmap: Shows relationships between features, highlighting redundancy or collinearity

Confusion Matrix Heatmap: Visualizes correct vs incorrect predictions

Actual vs Predicted Scatter Plot: Compares predicted values to true labels

📂 Repository Contents
Dataset_spine.csv – Dataset file

Spine_Linear_Regression.ipynb – Notebook with code, metrics, and visualizations

README.md – Project documentation and instructions

Visualizations/ – Folder containing pre-generated plots (optional)

🏃 Instructions to Run
Open Spine_Linear_Regression.ipynb in Google Colab or Jupyter Notebook

Upload Dataset_spine.csv

Run all cells to:

Load and preprocess data

Train the Linear Regression model

Evaluate model performance

View tables, metrics, and visualizations

Tip: Adjust the classification threshold to see its effect on accuracy and confusion matrix outcomes.

🔬 Potential Improvements
Use Logistic Regression, Decision Trees, or Random Forests for improved classification

Apply feature scaling or regularization for numerical stability

Perform cross-validation for more robust model evaluation

Analyze feature importance to identify influential features

Test ensemble methods for higher accuracy

🎯 Conclusion
This project demonstrates how Linear Regression can be adapted for classification tasks in medical data analysis. Through preprocessing, training, evaluation, and visualization, it provides insights into spinal condition prediction. The workflow can serve as a baseline for future models or academic exercises in healthcare-focused machine learning projects.

👨‍💻 Author
Name: Muhammad Haseeb Khan

University: Capital University of Science & Technology (CUST)

GitHub: haseebkhan17

Contact: your.email@example.com
