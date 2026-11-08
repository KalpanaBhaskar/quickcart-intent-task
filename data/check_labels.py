import pandas as pd

# Load the dataset and get all unique intent labels
df = pd.read_csv('data/train.csv')
intent_labels = df['intent'].unique().tolist()

print(intent_labels)
