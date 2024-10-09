import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# loading dataset
df = pd.read_csv("data/dataset.csv")

# data preview
print(df.head())

# check for missing values
print(df.isnull().sum())

# filling gaps with average values
df.fillna(df.mean(), inplace=True)

# descriptive statistics
print(df.describe())

# visualization of distribution of the target variable
sns.countplot(x="target", data=df)
plt.title("Distribution of Target Variable")
plt.show()

# correlation matrix
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# definition of feature and target variable
X = df.drop("target", axis=1)
y = df["target"]

# separation into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# training the model
model = LogisticRegression()
model.fit(X_train, y_train)

# prediction
y_pred = model.predict(X_test)

# accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# construction of the mixing matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
