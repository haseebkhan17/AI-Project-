# 🦴 Spine Condition Classification using Linear Regression

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-green) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 🌟 Project Overview

Spinal disorders are a significant health concern worldwide, often leading to pain, reduced mobility, and impaired quality of life. Early diagnosis and monitoring are crucial for effective treatment. This project leverages **machine learning techniques** to classify spinal conditions using a **biomechanical dataset of the human spine**.

A **Linear Regression** model is trained on biomechanical features and adapted for **binary classification** by applying a threshold on predicted values. The main goal is to **distinguish between normal and abnormal spinal conditions** and evaluate the model’s performance using quantitative metrics and visual visualizations. This serves as a strong academic example for applying regression techniques to medical data.

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

Data preprocessing is a crucial step to ensure the model receives **clean and structured data**. In this project, the following steps were applied:

1. **Missing Value Check**: Verified that no null or missing values existed.  
2. **Label Encoding**: Converted the `Class_att` target variable into numeric format.  
3. **Data Splitting**: Divided the dataset into **training (75%)** and **testing (25%)** sets.  
4. **Feature Selection**: All biomechanical features were selected for model training.  
5. **Normalization/Scaling (Optional)**: While not strictly necessary for Linear Regression, scaling can improve numerical stability for models in future iterations.  

---

## 🎯 Features and Target Definition

- **Features (X):** All spinal measurements (`Pelvic Incidence`, `Pelvic Tilt`, `Lumbar Lordosis Angle`, `Sacral Slope`, `Pelvic Radius`, `Degree of Spondylolisthesis`)  
- **Target (y):** `Class_att` (Binary: 0 = Normal, 1 = Abnormal)

This separation ensures a **supervised learning workflow**, where the model learns patterns in the features to predict the target class.

---

## 🛠 Model Training

- **Algorithm Used:** Linear Regression  
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`  
- **Training Process:**  
  1. Model fitted on the **training data** using all features.  
  2. Predictions generated for the **test set**.  
  3. Predictions converted to **binary classes** using a threshold (e.g., 0.5).  

**Why Linear Regression for Classification?**  
Linear Regression predicts continuous outcomes, but by applying a **threshold**, we can classify outputs into discrete categories. While not as robust as logistic regression for classification, it provides a **simple baseline model** and helps understand feature influence.

---

## 📏 Evaluation Metrics

The model was evaluated using metrics appropriate for classification tasks:

- **Accuracy:** Proportion of correct predictions.  
- **Confusion Matrix:** True positives, true negatives, false positives, and false negatives visualized.  
- **Scatter Plots:** Compare actual vs predicted outcomes.  

These metrics provide insight into the **model’s predictive power** and help identify areas for improvement.

---

## 📈 Visualizations

Visualizations make it easier to interpret and communicate model results:

1. **Feature Correlation Heatmap**: Shows relationships between features, highlighting redundancy or collinearity.  
2. **Confusion Matrix Heatmap**: Quickly visualizes correct vs incorrect predictions.  
3. **Actual vs Predicted Scatter Plot**: Compares predicted values to true labels, illustrating model performance.  

Sample visualization could look like:

Actual 0 | Predicted 0 -> True Negative
Actual 1 | Predicted 0 -> False Negative
Actual 0 | Predicted 1 -> False Positive
Actual 1 | Predicted 1 -> True Positive
---

## 📂 Repository Contents

- **Dataset file:** `Dataset_spine.csv`  
- **Notebook:** `Spine_Linear_Regression.ipynb`  
- **README:** Project documentation and instructions  
- **Visualizations Folder (Optional):** Pre-generated plots  

---

## 🏃 Instructions to Run

1. Open `Spine_Linear_Regression.ipynb` in **Google Colab** or **Jupyter Notebook**.  
2. Upload `Dataset_spine.csv`.  
3. Run all cells to:  
   - Load and preprocess data  
   - Train the Linear Regression model  
   - Evaluate model performance  
   - View tables, metrics, and visualizations  

> **Tip:** Adjust the classification threshold to see how it impacts accuracy and confusion matrix outcomes.

---

## 🔬 Potential Improvements

- Use **Logistic Regression** or **Decision Trees** for improved classification.  
- Apply **feature scaling** or **regularization** to improve model stability.  
- Perform **cross-validation** to better estimate model performance.  
- Explore **feature importance** to identify most influential biomechanical measurements.  
- Test **ensemble methods** for higher accuracy on medical datasets.

---

## 🎯 Conclusion

This project demonstrates how **Linear Regression can be adapted for classification tasks** in medical data analysis. Through preprocessing, training, evaluation, and visualization, it provides insights into **spinal condition prediction**. The workflow can serve as a **baseline for future models** or academic exercises in healthcare-focused machine learning projects.

---

## 👨‍💻 Author

- **Name:** Muhammad Haseeb Khan  
- **University:** Capital University of Science & Technology (CUST)  

---

