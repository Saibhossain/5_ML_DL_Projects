import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments


import kagglehub
import pandas as pd
import os

# Download latest version
path = kagglehub.dataset_download("mayaaboelkhier/amazon-reviews-multilingual-us-v1-00")

# Function to find the TSV file recursively in subdirectories
def find_tsv_file(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.tsv'):
                return os.path.join(root, file)
    return None  # Return None if no TSV file is found

# Find the TSV file
tsv_path = find_tsv_file(path)

# Read the TSV file if found
if tsv_path:
    df = pd.read_csv(tsv_path, delimiter='\t', on_bad_lines='skip')  # Use delimiter='\t' for TSV files
    df.head()
    print(df)
    print("Path to dataset files:", tsv_path)
else:
    print("No TSV file found in the downloaded dataset.")

print(df.info())
print(df.describe())

# Map star rating to sentiment
def map_sentiment(rating):
    if rating <= 2:
        return 0  # Negative
    elif rating == 3:
        return 1  # Neutral
    else:
        return 2  # Positive

df = df[['review_body', 'star_rating']].dropna()
print(df.head())
df = df[df['review_body'].apply(lambda x: isinstance(x, str))]
print(df.head())

df['sentiment'] = df['star_rating'].apply(map_sentiment)

# Check class distribution
df['sentiment'].value_counts()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenizer_function(example):
  return tokenizer(example['review_body'], padding='max_length', truncation=True)

from datasets import Dataset
dataset = Dataset.from_pandas(df[['review_body', 'sentiment']])
dataset = dataset.map(tokenizer_function, batched=True)
dataset = dataset.rename_column('sentiment', 'labels')
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

train_test = dataset.train_test_split(test_size=0.2, seed=42)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_test['train'],
    eval_dataset=train_test['test'],
)
trainer.train()
preds_output = trainer.predict(train_test['test'])
y_pred = np.argmax(preds_output.predictions, axis=1)
y_true = preds_output.label_ids

print(classification_report(y_true, y_pred, target_names=["Negative", "Neutral", "Positive"]))
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    label = torch.argmax(probs).item()
    return ["Negative", "Neutral", "Positive"][label]

# Test
predict_sentiment("The product is amazing and works perfectly!")