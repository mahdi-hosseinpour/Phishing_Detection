import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import numpy as np
import torch
import re
import urllib.parse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding

def preprocess(url):
    url = str(url).lower()
    url = urllib.parse.unquote(url)
    p = urllib.parse.urlparse(url)
    domain = re.sub(r'^www\.', '', p.netloc.split(':')[0])
    path = p.path or "/"
    query = "?" + p.query if p.query else ""
    return f"[DOMAIN] {domain} [PATH] {path} [QUERY] {query}"

print("Loading dataset...")
df = pd.read_csv('data/malicious_phish.csv')[['url', 'type']].dropna()

print("Initial class distribution:")
print(df['type'].value_counts())

# Split the data
train_val, test = train_test_split(df, test_size=0.15, random_state=42, stratify=df['type'])
train, val = train_test_split(train_val, test_size=0.15, random_state=42, stratify=train_val['type'])

# Balance the training set (reduce benign samples)
benign = train[train['type'] == 'benign']
others = train[train['type'] != 'benign']
benign_downsampled = benign.sample(n=len(others)//3 + 12000, random_state=42, replace=False)
train_balanced = pd.concat([benign_downsampled, others]).sample(frac=1, random_state=42)

# Label encoding
le = LabelEncoder()
train_balanced['label'] = le.fit_transform(train_balanced['type'])
val['label'] = le.transform(val['type'])
test['label'] = le.transform(test['type'])

# Preprocess URLs
train_balanced['text'] = train_balanced['url'].apply(preprocess)
val['text'] = val['url'].apply(preprocess)
test['text'] = test['url'].apply(preprocess)

# Fix index issue that causes problems with Hugging Face datasets
train_balanced = train_balanced.reset_index(drop=True)
val = val.reset_index(drop=True)
test = test.reset_index(drop=True)

# Create Hugging Face DatasetDict
dataset = DatasetDict({
    'train': Dataset.from_pandas(train_balanced[['text', 'label']], preserve_index=False),
    'val': Dataset.from_pandas(val[['text', 'label']], preserve_index=False),
    'test': Dataset.from_pandas(test[['text', 'label']], preserve_index=False)
})

# Tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("jackaduma/SecBERT")

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, max_length=256)

tokenized = dataset.map(tokenize, batched=True)
tokenized = tokenized.remove_columns(['text'])
tokenized = tokenized.rename_column('label', 'labels')
tokenized.set_format('torch')

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model = AutoModelForSequenceClassification.from_pretrained("jackaduma/SecBERT", num_labels=4)

# Compute metrics (accuracy + F1) during training
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

# Training arguments with frequent evaluation
args = TrainingArguments(
    output_dir="./secbert_result",
    num_train_epochs=6,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=1,
    learning_rate=2e-5,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=20,
    eval_steps=500,                    # Evaluate every 500 steps
    evaluation_strategy="steps",       # Evaluate by steps instead of epochs
    save_strategy="steps",
    save_steps=1000,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",  # Keep the model with highest accuracy
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=2,
    report_to="none",
    seed=42,
    disable_tqdm=False,
    logging_dir='./logs',
    logging_first_step=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['val'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,   # This enables accuracy/F1 logging
)

trainer.train()

# Final evaluation on test set
print("\nFinal evaluation on test set...")
preds = trainer.predict(tokenized['test'])
y_pred = np.argmax(preds.predictions, axis=1)
print("\n" + classification_report(preds.label_ids, y_pred, target_names=le.classes_, digits=6))

# Save the final model
trainer.save_model("./final_phishing_model_995")
tokenizer.save_pretrained("./final_phishing_model_995")
print("\nFinal model saved successfully!")