# 📢 K-Nearest Neighbors (KNN) Classification Project

This Jupyter Notebook implements a **K-Nearest Neighbors (KNN)** classification model to predict a **binary target class** from an unknown dataset. The notebook includes **Exploratory Data Analysis (EDA)**, feature preprocessing, model training, and evaluation using popular Python libraries.

The goal is to determine whether a data point belongs to one of two categories based on its features using the **KNN algorithm**, and to find the best value of **k** that gives the most accurate predictions.

## 📊 Dataset

This dataset contains numerical features which are labeled and a binary `TARGET CLASS` column indicating the outcome or category for each observation.

🔗 [Download the dataset from Kaggle](https://www.kaggle.com/datasets/nitya1510/knn-project-data )

## 🔍 Features

✅ Exploratory Data Analysis (EDA) with visualizations  
✅ Feature scaling using `StandardScaler` for better KNN performance  
✅ Train/Test split and model training using `KNeighborsClassifier`  
✅ Evaluation metrics: Confusion matrix, classification report, accuracy score  
✅ Hyperparameter tuning: Using the Elbow Method to select the optimal `k` value  
💡 Insights into how different features influence the prediction of the target class  

## 🛠️ How to Use

1. Clone this repository
2. Download the dataset and place it in the project folder
3. Install required packages: `pandas`, `numpy`, `sklearn`, `matplotlib`, and `seaborn`
4. Open the notebook in Jupyter Notebook
5. Run the cells to:
   - Explore the data visually
   - Preprocess and scale the features
   - Train and evaluate the KNN model
   - Tune `k` values using error rate analysis

## 🧠 Tips & Notes

- KNN is sensitive to feature scales, so we use `StandardScaler` before modeling
- The Elbow Method helps choose the best `k` by minimizing error
- This is a great example of working with real-world unlabeled datasets

## 👨‍💻 Author

Krish Makwana