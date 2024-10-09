import pandas as pd


df = pd.read_csv("data/dataset.csv")

print(df.head())

print(df.isnull().sum())

df.fillna(df.mean(), inplace=True)

