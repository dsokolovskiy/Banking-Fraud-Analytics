import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# Load the dataset
df = pd.read_csv("data/dataset.csv")

# Display data types in the DataFrame
print("Data types in DataFrame:")
print(df.dtypes)

# Display descriptive statistics
print(df.describe(include='all'))

# Display the distribution of the target variable
target_counts = df['target'].value_counts()
print("Distribution of target variable:")
print(target_counts)

# Split the dataset into training and test sets
X = df.drop(['target', 'id', 'sample_type'], axis=1)  # Features
y = df['target']  # Target variable

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Create and display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Display feature importances
importance = model.feature_importances_
feature_names = X.columns

# Ensure that the lengths match before creating the DataFrame
if len(importance) == len(feature_names):
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Plotting feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()
else:
    raise ValueError("The lengths of importances and feature names do not match.")

# Visualizing the distribution of the target variable
sns.countplot(data=df, x='target')
plt.title('Distribution of Target Variable')
plt.show()
