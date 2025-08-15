💻 Laptop Price Prediction

This project predicts laptop prices based on their specifications such as brand, CPU, RAM, storage, screen resolution, GPU, operating system, and weight.
It uses Python, Scikit-learn, Gradient Boosting Regressor, and feature engineering techniques to build a deployable prediction pipeline.

📌 Features

Data Cleaning & Preprocessing: Handles missing values, encodes categorical variables, scales numeric features.

Feature Engineering: Extracts CPU brand/type, GPU type (integrated/dedicated), screen resolution details, and storage type.

Model Training: Uses Gradient Boosting Regressor with hyperparameter tuning via RandomizedSearchCV.

Model Evaluation: Calculates R² Score, MAE, and RMSE for budget and premium laptop segments.

Deployment Ready: Model, encoders, and scaler are saved with joblib for easy integration into APIs or web apps.

📊 Dataset

The dataset contains details of laptops including brand, type, inches, screen resolution, CPU, RAM, memory, GPU, operating system, and weight.

The target variable is Price (log-transformed during training).

You can replace the dataset with your own CSV file following the same structure.

🛠 Tech Stack

Languages: Python

Libraries: Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib

Deployment: Joblib for model saving, Flask/FastAPI-ready pipeline
