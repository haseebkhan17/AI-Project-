# 🦴 Spine Condition Classification using Linear Regression

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-green?style=flat-square)
![GitHub](https://img.shields.io/badge/GitHub-Portfolio-lightgrey?style=flat-square)

---

## 🌟 Project Overview

Spinal disorders are a significant health concern worldwide, often leading to pain, reduced mobility, and impaired quality of life. Early diagnosis and monitoring are crucial for effective treatment.  

This project leverages **machine learning techniques** to classify spinal conditions using a **biomechanical dataset of the human spine**.  

A **Linear Regression** model is trained on biomechanical features and adapted for **binary classification** by applying a threshold on predicted values. The main goal is to **distinguish between normal and abnormal spinal conditions** and evaluate the model using metrics and visualizations.  

This project serves as a **strong academic example** for applying regression techniques to medical datasets in healthcare.

---

## 📊 Dataset Description

| Attribute | Description |
|-----------|-------------|
| **Dataset Name** | Spine Dataset |
| **Source** | [Kaggle](https://www.kaggle.com) |
| **Rows / Columns** | 310 rows, 6 features + 1 target |

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

**Observation:** All features are numeric and suitable for **regression-based modeling**.

---

## 🧹 Data Preprocessing

1. Verified **no missing/null values** exist.  
2. Converted `Class_att` target variable to numeric format.  
3. Split dataset into **training (75%)** and **testing (25%)** sets.  
4. Selected all biomechanical features for model training.  
5. Optional: **Feature scaling/normalization** for future numerical stability.

---

## 🎯 Features & Target

- **Features (X):** Pelvic Incidence, Pelvic Tilt, Lumbar Lordosis Angle, Sacral Slope, Pelvic Radius, Degree of Spondylolisthesis  
- **Target (y):** Class_att (Binary: 0 = Normal, 1 = Abnormal)

---

## 🛠 Model Training

- **Algorithm:** Linear Regression  
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`  

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize and train Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict and convert to binary
y_pred = lr.predict(X_test)
y_pred_binary = [1 if val >= 0.5 else 0 for val in y_pred]
Why Linear Regression for Classification?
Linear Regression predicts continuous values; applying a threshold allows classification. This provides a simple baseline model and insight into feature influence.

📏 Evaluation Metrics
Accuracy: Proportion of correct predictions

Confusion Matrix: Visualizes true positives, true negatives, false positives, false negatives

Scatter Plots: Compare actual vs predicted outcomes

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
📈 Visualizations
Feature Correlation Heatmap: Identify relationships or collinearity between features

Confusion Matrix Heatmap: Visualize correct vs incorrect predictions

Actual vs Predicted Scatter Plot: Compare predictions with true labels

These visualizations make model performance intuitive and easy to interpret.

📂 Repository Contents
File	Description
Dataset_spine.csv	Dataset file
Spine_Linear_Regression.ipynb	Notebook with code, metrics, visualizations
README.md	Project documentation and instructions
Visualizations/	Folder containing pre-generated plots (optional)

🏃 Instructions to Run
Open Spine_Linear_Regression.ipynb in Google Colab or Jupyter Notebook.

Upload Dataset_spine.csv.

Run all cells to:

Load and preprocess data

Train Linear Regression

Evaluate metrics and view visualizations

Tip: Experiment with the threshold to observe its effect on accuracy and confusion matrix.

🔬 Potential Improvements
Switch to Logistic Regression, Decision Trees, or Random Forests

Apply feature scaling or regularization for numerical stability

Perform cross-validation for robust model evaluation

Analyze feature importance to identify influential measurements

Explore ensemble methods for higher predictive accuracy

🎯 Conclusion
This project demonstrates how Linear Regression can be adapted for classification in medical data analysis. The workflow provides insights into spinal condition prediction and serves as a baseline for healthcare machine learning projects.

👨‍💻 Author
Name: Haseeb Khan

University: Capital University of Science & Technology (CUST)

GitHub: haseebkhan17

Contact: [Your Email or GitHub Link]
