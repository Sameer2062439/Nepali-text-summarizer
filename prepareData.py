# import pandas as pd

# # Read the data from CSV file
# print("Loading data...")
# data = pd.read_csv('../dataset/updatedCSV.csv', encoding='utf-8')

# # Remove the "author" column
# data = data.drop("author", axis=1)

# print(len(data))

# # Drop NaN and empty rows
# data.dropna(inplace=True)

# # Remove rows with null values in "title" or "content" columns
# data = data.dropna(subset=["title", "content"])

# print(len(data))

# # Save the modified DataFrame to CSV
# data.to_csv('newsData.csv', index=False)

# -*- coding: utf-8 -*-

import pandas as pd

# Read the JSONL file
df = pd.read_json('nepali_train.jsonl', lines=True)

# Remove 'id' and 'url' columns
df = df.drop(['id', 'url'], axis=1)

# Decrease the length of rows to 2998
df = df.iloc[:80]

# Save the updated dataframe back to JSONL
df.to_json('nepali_train_80.jsonl', orient='records', lines=True)