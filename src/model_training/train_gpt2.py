import torch
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
import json
from pathlib import Path
from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger("GPT-2 Trainer")

class MisinfoDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_data(file_path):
    claims = []
    labels = []
    with open(file_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            try:
                data = json.loads(line)
                claims.append(data["claim"])
                labels.append(1 if data["label"] in ["true", "real"] else 0)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in {file_path} at line {line_num}: {line}")
                raise  # Optional: Remove this to continue processing despite errors
    return claims, labels

def main():
    # Load preprocessed data
    train_claims, train_labels = load_data("data/processed/liar.jsonl")
    val_claims, val_labels = load_data("data/processed/fakenewsnet.jsonl")

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize claims
    train_encodings = tokenizer(
        train_claims, 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )
    val_encodings = tokenizer(
        val_claims, 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )

    # Create datasets
    train_dataset = MisinfoDataset(train_encodings, train_labels)
    val_dataset = MisinfoDataset(val_encodings, val_labels)

    # Model setup
    model = GPT2ForSequenceClassification.from_pretrained(
        "gpt2",
        num_labels=2,
        pad_token_id=tokenizer.eos_token_id
    )
    model.config.pad_token_id = model.config.eos_token_id

    # Training arguments
    training_args = TrainingArguments(
        output_dir="../results/gpt2",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        evaluation_strategy="epoch",
        logging_dir="../logs/gpt2",
        learning_rate=5e-5,
        weight_decay=0.01,
        save_strategy="epoch"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train and save
    logger.info("Starting GPT-2 training...")
    trainer.train()
    model.save_pretrained(Config.MODEL_SAVE_PATH)
    tokenizer.save_pretrained(Config.MODEL_SAVE_PATH)
    logger.info(f"Model saved to {Config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()