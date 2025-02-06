import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from src.utils.config import Config
from src.utils.logger import setup_logger
import gc

# Initialize logger
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
    # Connect to MongoDB and fetch data
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

    # Tokenization with padding, truncation and batching
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Tokenize training data
    logger.info("Tokenizing training data...")
    train_encodings = tokenizer(
        train_texts, 
        truncation=True, 
        padding='max_length', 
        max_length=64,  # Reduced max length
        return_tensors='pt'  # This will return tensors directly
    )

    # Tokenize validation data
    logger.info("Tokenizing validation data...")
    val_encodings = tokenizer(
        val_texts, 
        truncation=True, 
        padding='max_length', 
        max_length=64,  # Reduced max length
        return_tensors='pt'  # This will return tensors directly
    )

    # Create datasets
    train_dataset = ClaimDataset(train_encodings, train_labels)
    val_dataset = ClaimDataset(val_encodings, val_labels)

    # Create DataLoader for batch processing
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Reduced batch size
    val_loader = DataLoader(val_dataset, batch_size=8)

    # Model setup
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=3
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir='../results/bert',
        num_train_epochs=3,
        per_device_train_batch_size=8,  # Reduced batch size
        evaluation_strategy='epoch',
        logging_dir='../logs/bert',
        save_strategy='epoch',
        load_best_model_at_end=True,
        logging_steps=10,  # Logs after every 10 steps
        fp16=True,  # Enable mixed precision
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=None,  # Batching is handled by DataLoader
    )

    # Start training and save model
    logger.info("Starting BERT training...")
    trainer.train()
    model.save_pretrained(Config.MODEL_SAVE_PATH)
    tokenizer.save_pretrained(Config.MODEL_SAVE_PATH)
    logger.info(f"BERT model saved to {Config.MODEL_SAVE_PATH}")

    # Clean up resources
    gc.collect()

if __name__ == '__main__':
    main()
