import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score, classification_report
from datasets import Dataset, DatasetDict
from transformers import BertTokenizerFast, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import re
import urllib.parse

# ------------------- 2. Load Dataset -------------------
df = pd.read_csv('data/phishing.csv')  # Adjust the file name if needed
df = df[['url', 'type']].dropna()

print("Initial class distribution:")
print(df['type'].value_counts())

# ------------------- 3. Stratified Train/Val/Test Split -------------------
train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['type'])
train_df, val_df = train_test_split(train_val_df, test_size=0.15, random_state=42, stratify=train_val_df['type'])

print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ------------------- 4. Balance ONLY the Train set with Undersampling -------------------
print("\nBalancing the training set...")

benign = train_df[train_df['type'] == 'benign']
malware = train_df[train_df['type'] == 'malware']
phishing = train_df[train_df['type'] == 'phishing']
defacement = train_df[train_df['type'] == 'defacement']

# Slightly oversample benign to be close to the largest minority class
target_samples = max(len(malware), len(phishing), len(defacement)) + 1000

benign_balanced = resample(benign, replace=False, n_samples=target_samples, random_state=42)

train_balanced = pd.concat([benign_balanced, malware, phishing, defacement])
train_balanced = train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("Class distribution in train after balancing:")
print(train_balanced['type'].value_counts())

# ------------------- 5. Label Encoding -------------------
label_encoder = LabelEncoder()
train_balanced['label'] = label_encoder.fit_transform(train_balanced['type'])
val_df['label'] = label_encoder.transform(val_df['type'])
test_df['label'] = label_encoder.transform(test_df['type'])

label2id = {l: i for i, l in enumerate(label_encoder.classes_)}
id2label = {i: l for l, i in label2id.items()}
num_labels = len(label2id)

print(f"\nLabel mapping: {label2id}")

# ------------------- 6. Advanced URL Preprocessing -------------------
def advanced_url_preprocess(url):
    url = url.lower()
    url = urllib.parse.unquote(url)
    parsed = urllib.parse.urlparse(url)
    domain = re.sub(r'^www\.', '', parsed.netloc)
    path = parsed.path if parsed.path else "/"
    query = parsed.query if parsed.query else ""
    return f"[DOMAIN] {domain} [PATH] {path} [QUERY] {query}".strip()

train_balanced['text'] = train_balanced['url'].apply(advanced_url_preprocess)
val_df['text'] = val_df['url'].apply(advanced_url_preprocess)
test_df['text'] = test_df['url'].apply(advanced_url_preprocess)

# ------------------- 7. Convert to Hugging Face Dataset -------------------
train_ds = Dataset.from_pandas(train_balanced[['text', 'label']])
val_ds = Dataset.from_pandas(val_df[['text', 'label']])
test_ds = Dataset.from_pandas(test_df[['text', 'label']])

dataset = DatasetDict({'train': train_ds, 'validation': val_ds, 'test': test_ds})

# ------------------- 8. Tokenizer -------------------
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(['text'])
tokenized_dataset = tokenized_dataset.rename_column('label', 'labels')
tokenized_dataset.set_format('torch')

# ------------------- 9. Data Collator -------------------
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ------------------- 10. Compute Class Weights -------------------
from sklearn.utils.class_weight import compute_class_weight
classes = np.arange(num_labels)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_balanced['label'])
class_weights = torch.tensor(weights, dtype=torch.float).to('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Class Weights: {class_weights}")

# ------------------- 11. Custom Trainer with Weighted Loss -------------------
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(model.device))
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    f1_per_class = f1_score(labels, preds, average=None)
    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_benign': f1_per_class[label2id['benign']],
        'f1_phishing': f1_per_class[label2id['phishing']],
        'f1_malware': f1_per_class[label2id['malware']],
        'f1_defacement': f1_per_class[label2id['defacement']],
    }

# ------------------- 12. Model & Trainer Setup -------------------
model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label
)

training_args = TrainingArguments(
    output_dir='./url_malware_bert',
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=4,        # Effective batch size = 16 × 4 = 64
    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_steps=100,
    logging_steps=100,

    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    save_total_limit=2,

    report_to="none",
    seed=42,
    remove_unused_columns=False,
    dataloader_num_workers=2,
    dataloader_pin_memory=False,

    fp16=False,                    # Set to False on CPU
    gradient_checkpointing=True,   # Saves ~50% VRAM – highly recommended
    optim="adamw_torch",
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ------------------- 13. Training -------------------
print("\nTraining started...")
trainer.train()

# ------------------- 14. Final Evaluation on Test Set -------------------
print("\nFinal evaluation on the test set:")
test_results = trainer.evaluate(tokenized_dataset['test'])
print(test_results)

# Predictions and detailed report
predictions = trainer.predict(tokenized_dataset['test'])
preds = np.argmax(predictions.predictions, axis=1)
print("\nFinal classification report on test set:")
print(classification_report(predictions.label_ids, preds, target_names=label_encoder.classes_))

# ------------------- 15. Save Model -------------------
trainer.save_model('./final_url_bert_model')
tokenizer.save_pretrained('./final_url_bert_model')
print("\nModel saved to './final_url_bert_model'")