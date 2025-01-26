## 🏭 Industrial Copper Modeling
## 📘 Overview
This project focuses on leveraging machine learning to predict the selling price and status (e.g., won/lost) of copper in the manufacturing industry. With clean data and advanced modeling techniques, this solution aims to streamline copper sales predictions while addressing data irregularities, such as missing values, skewness, and outliers.

## 🔑 Key Features
Predict Selling Price: A Random Forest Regression model predicts the selling price of copper with high accuracy.
Predict Status: An Extra Trees Classification model determines whether a copper deal will be won or lost.
Streamlit Integration: An interactive app enables users to input data and get instant predictions.
Advanced Preprocessing:
Handling missing values using median/mode.
Removing outliers with IQR (Interquartile Range).
Addressing skewness using log transformations.
Deployment Ready: Models saved as .pkl files for seamless use in applications.

## 📚 Technologies and Tools
Programming Language: Python 🐍
Libraries:
Data Manipulation: pandas, numpy
Visualization: matplotlib, seaborn, plotly
Machine Learning: scikit-learn, xgboost, imblearn
Deployment: pickle, Streamlit
Machine Learning Techniques:
Random Forest, Extra Trees, and Gradient Boosting.
Hyperparameter tuning using GridSearchCV.
Data resampling with SMOTETomek for balancing imbalanced datasets.

## ⚙️ Project Workflow
Data Preprocessing:

Loaded and cleaned the dataset.
Filled missing values using median and mode.
Addressed skewness and removed outliers with IQR.
Encoded categorical features for machine learning.
Feature Engineering:

Correlation analysis with heatmaps.
Derived features, including time differences and logs, for better predictions.
Modeling:

Classification:
Built a Random Forest Classifier to predict copper deal status.
Resolved imbalanced data issues with SMOTETomek.
Regression:
Developed a Random Forest Regression model to predict copper selling prices.
Fine-tuned models using GridSearchCV.
Evaluation:
Used metrics like accuracy, R², RMSE, and ROC-AUC for evaluation.
Deployment:

Pickled trained models for easy reuse.
Streamlit app for user interaction and prediction.

## 🌟 How to Use
Clone this repository:
git clone <repository_link>
cd industrial-copper-modeling
Install the required libraries:

pip install pandas numpy scikit-learn xgboost matplotlib seaborn plotly streamlit imbalanced-learn
Run the Streamlit app:
bash
Copy
Edit
streamlit run app.py
Input the data into the app, and get predictions for:
Selling Price (Regression)
Deal Status (Classification)

## 💻 Example Predictions
Classification (Status)
[77.0, 3.0, 10.0, 1500.0, 164141591, 3.68, 17.22, 0.0, 7.11, 1, 4, 2021, 1, 8, 2021]
Output: Status: Won

Regression (Selling Price)
[30202938, 25, 1, 5, 41, 1210, 1668701718, 6.6, -0.2, 1, 4, 2021, 1, 4, 2021]
Output:


Edit
Predicted Selling Price (Log): 6.79
Predicted Selling Price: ₹900

## 🎯 Project Highlights
Real-world Application: Automates predictions for critical business metrics.
Scalable: Ready for integration into larger systems with its modular architecture.
User-friendly: Equipped with an interactive UI for non-technical stakeholders.

## 📊 Key Metrics
Classification Accuracy: >90%
Regression R² Score: >0.85
Balanced Datasets: Achieved with SMOTETomek.

## 👨‍🏫 References
Python Documentation
pandas Documentation
scikit-learn Documentation
numpy Documentation
Streamlit Documentation
