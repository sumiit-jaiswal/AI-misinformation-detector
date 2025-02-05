from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from pymongo import MongoClient
from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger("Sarcasm Trainer")

class SarcasmDataset(Dataset):
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
    sarcasm_data = list(db.sarcasm_data.find({}))
    
    # Prepare data
    texts = [item['text'] for item in sarcasm_data]
    labels = [item['label'] for item in sarcasm_data]  # 0=genuine, 1=sarcastic

    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained("sismetanin/sarcasm-detection")
    
    encodings = tokenizer(
        texts, 
        truncation=True, 
        padding='max_length', 
        max_length=128
    )

    # Split dataset
    train_encodings = {k: v[:int(0.8*len(v))] for k, v in encodings.items()}
    val_encodings = {k: v[int(0.8*len(v)):] for k, v in encodings.items()}
    train_labels = labels[:int(0.8*len(labels))]
    val_labels = labels[int(0.8*len(labels)):]

    # Create datasets
    train_dataset = SarcasmDataset(train_encodings, train_labels)
    val_dataset = SarcasmDataset(val_encodings, val_labels)

    # Model setup
    model = AutoModelForSequenceClassification.from_pretrained(
        "sismetanin/sarcasm-detection",
        num_labels=2
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir='../results/sarcasm',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        evaluation_strategy='epoch',
        logging_dir='../logs/sarcasm'
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train and save
    logger.info("Starting sarcasm model training...")
    trainer.train()
    model.save_pretrained(Config.SARCASM_MODEL_PATH)
    logger.info(f"Sarcasm model saved to {Config.SARCASM_MODEL_PATH}")

if __name__ == '__main__':
    main()