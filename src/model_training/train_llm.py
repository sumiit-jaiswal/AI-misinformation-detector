import torch
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from src.utils.config import Config
from src.utils.logger import setup_logger
from pymongo import MongoClient
import numpy as np
from sklearn.model_selection import train_test_split

logger = setup_logger("LLM Trainer")

# Initialize MongoDB connection
client = MongoClient(Config.MONGO_URI)
db = client[Config.DB_NAME]
claims = list(db.claims.find({}))

# Prepare dataset
texts = [claim["text"] for claim in claims]
labels = [claim["label"] for claim in claims]  # 0=False, 1=True, 2=Ambiguous

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Tokenization
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_encodings = tokenizer(
    train_texts, truncation=True, padding=True, max_length=128
)
val_encodings = tokenizer(
    val_texts, truncation=True, padding=True, max_length=128
)

class ClaimDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ClaimDataset(train_encodings, train_labels)
val_dataset = ClaimDataset(val_encodings, val_labels)

# Model setup
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=3
)

training_args = TrainingArguments(
    output_dir="../results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    logging_dir="../logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train and save
logger.info("Starting LLM training...")
trainer.train()
model.save_pretrained(Config.MODEL_SAVE_PATH)
tokenizer.save_pretrained(Config.MODEL_SAVE_PATH)
logger.info(f"Model saved to {Config.MODEL_SAVE_PATH}")