import torch
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
from pymongo import MongoClient
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger("BERT Trainer")

class ClaimDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)

def main():
    # Connect to MongoDB
    client = MongoClient(Config.MONGO_URI)
    db = client[Config.DB_NAME]
    claims = list(db.claims.find({}))
    
    # Prepare data
    texts = [claim['text'] for claim in claims]
    labels = [claim['label'] for claim in claims]  # 0=False, 1=True, 2=Ambiguous
    
    # Split dataset
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Tokenization
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    train_encodings = tokenizer(
        train_texts, 
        truncation=True, 
        padding='max_length', 
        max_length=128
    )
    
    val_encodings = tokenizer(
        val_texts, 
        truncation=True, 
        padding='max_length', 
        max_length=128
    )

    # Create datasets
    train_dataset = ClaimDataset(train_encodings, train_labels)
    val_dataset = ClaimDataset(val_encodings, val_labels)

    # Model setup
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=3
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir='../results/bert',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        evaluation_strategy='epoch',
        logging_dir='../logs/bert',
        save_strategy='epoch'
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train and save
    logger.info("Starting BERT training...")
    trainer.train()
    model.save_pretrained(Config.MODEL_SAVE_PATH)
    tokenizer.save_pretrained(Config.MODEL_SAVE_PATH)
    logger.info(f"BERT model saved to {Config.MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()