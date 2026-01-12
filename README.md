# AI-Project-
This project uses Linear Regression to analyze a spine dataset and perform binary classification. The model is evaluated using accuracy, a confusion matrix, and basic visualizations, and is implemented in Python using Google Colab.

# 🦴 Spine Condition Classification using Linear Regression

## 🌟 Project Overview

This project applies machine learning to classify spinal conditions using a medical spine dataset. A Linear Regression model is trained on biomechanical features of the human spine and adapted for binary classification by applying a threshold on predicted values. The goal is to distinguish between normal and abnormal spinal conditions and evaluate the model’s performance using standard metrics and visualizations.

## 📊 Dataset Description

**Dataset Name:** Spine Dataset  
**Source:** Kaggle  
**Number of Rows / Columns:** 310 rows, 6 features + 1 target  

**Features:**
- Pelvic Incidence
- Pelvic Tilt
- Lumbar Lordosis Angle
- Sacral Slope
- Pelvic Radius
- Degree of Spondylolisthesis

**Target Variable:**
- Class_att (0 = Normal, 1 = Abnormal)

**Observation:**  
All features are numeric and suitable for regression-based modeling.

## 🧹 Data Preprocessing

- Checked and verified missing values
- Converted class labels into numeric format
- Split dataset into training (75%) and testing (25%) sets
- Selected all biomechanical features for model training

## 🎯 Features and Target Definition

- **Features (X):** All spinal measurements  
- **Target (y):** Class_att

## 🛠 Model Training

- **Algorithm Used:** Linear Regression  
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn  
- Model trained on training data and used to predict test outcomes

## 📏 Evaluation Metrics

- Accuracy  
- Confusion Matrix  
- Regression-based predictions converted into binary classes

## 📈 Visualizations

- Feature correlation heatmap  
- Confusion matrix heatmap  
- Actual vs predicted scatter plot

## 📂 Repository Contents

- **Dataset file:** `Dataset_spine.csv`  
- **Notebook:** `Spine_Linear_Regression.ipynb`  
- **README:** Project documentation and instructions

## 🏃 Instructions to Run

1. Open the notebook in Google Colab  
2. Upload `Dataset_spine.csv`  
3. Run all cells to:  
   - Load and preprocess data  
   - Train the Linear Regression model  
   - View tables, metrics, and graphs

## 🎯 Conclusion

This project demonstrates how Linear Regression can be adapted for classification tasks in medical data analysis. Through evaluation metrics and visualizations, the model provides insights into spinal condition prediction and serves as a strong academic example of machine learning workflow.

## 👨‍💻 Author

- **Name:** Haseeb Khan  
- **University:** Capital University of Science & Technology (CUST)
