# Banking Fraud Analytics

## Project Description
This project aims to analyze and predict fraudulent transactions in the banking sector using machine learning techniques. The analysis involves data cleaning, feature engineering, model training, and evaluation to detect potential fraudulent activities.

## Installation
To set up the project, please follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dsokolovskiy/Banking-Fraud-Analytics
   cd BankingFraudAnalytics

2. Create a virtual environment:
    ```bash
   python -m venv .venv
3. Activate the virtual environment:
 - For Windows:
   ```bash 
   .venv\Scripts\activate
   ```
 - For macOS/Linux:
   ```bash 
   source .venv/bin/activate
   ```
4. Install the required packages:
   ```bash 
   pip install -r requirements.txt
   ```

# Usage
To run the data analysis and model training script, execute the following command:
   ```bash 
python data_analyses.py
```
The script will:
- Load the dataset
- Display data types and descriptive statistics
- Analyze the distribution of the target variable
- Train a Random Forest model
- Evaluate the model`s accuracy and display the confusion matrix
- Visualize feature importances and the distribution of the target variable

# Data Analysis
The data analysis includes:
- Overview of data types and descriptive statistics.
- Exploration of the target variable`s distribution.
- Training a Random Forest Classifier to predict fraudulent transactions.
- Evaluation metrics, including accuracy and confusion matrix.

# Model Evaluation
The model achieved an accuracy of 92% on the dataset. The confusion matrix indicates how well the model is performing in terms of true positives, false positives, true negatives, and false negatives.

# Recommendations
- Explore additional feature engineering techniques to improve model accuracy.
- Experiment with different machine learning models to find the best fit fot the dataset.
- Implement cross - validation for more reliable model evaluation.

