import torch
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
import json
import gc
from pathlib import Path
from src.utils.config import Config
from src.utils.logger import setup_logger

# Setup logger
logger = setup_logger("GPT-2 Trainer")

# Dataset class definition
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

# Data loading function
# Data loading function with additional logging
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
                continue  # Skip malformed lines instead of raising an exception
            except KeyError as e:
                logger.error(f"Missing key in JSON at line {line_num}: {line}")
                continue  # Skip lines that don't have the expected keys
    return claims, labels


def main():
    # Load preprocessed data (limit data size for testing)
    train_claims, train_labels = load_data("data/processed/liar.jsonl")[:500]  # Limit data for testing
    val_claims, val_labels = load_data("data/processed/fakenewsnet.jsonl")[:500]

    # Debugging: Check the number of claims loaded
    logger.info(f"Loaded {len(train_claims)} train claims.")
    logger.info(f"Loaded {len(val_claims)} validation claims.")

    # Check if the val_claims list is empty
    if not val_claims:
        logger.error("Validation claims list is empty. Exiting.")
        return

    # Optionally, print the first few claims to inspect data
    logger.info(f"First 3 train claims: {train_claims[:3]}")
    logger.info(f"First 3 validation claims: {val_claims[:3]}")

    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir="./cache")  # Use the regular GPT-2 model
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token for GPT-2

    # Tokenize claims
    logger.info("Tokenizing training data...")
    train_encodings = tokenizer(train_claims, truncation=True, padding="max_length", max_length=128)

    logger.info("Tokenizing validation data...")
    val_encodings = tokenizer(val_claims, truncation=True, padding="max_length", max_length=128)

    # Check if tokenization worked properly
    logger.info(f"Train encodings: {train_encodings.keys()}")
    logger.info(f"Validation encodings: {val_encodings.keys()}")

    # Create datasets
    train_dataset = MisinfoDataset(train_encodings, train_labels)
    val_dataset = MisinfoDataset(val_encodings, val_labels)

    # Model setup with regular GPT-2
    model = GPT2ForSequenceClassification.from_pretrained(
        "gpt2",  # Using regular GPT-2 model (not small)
        num_labels=2,
        pad_token_id=tokenizer.eos_token_id  # Set pad token ID to eos token ID
    )
    model.config.pad_token_id = model.config.eos_token_id

    # Training arguments with reduced batch size and gradient accumulation
    training_args = TrainingArguments(
        output_dir="../results/gpt2",
        num_train_epochs=3,
        per_device_train_batch_size=4,  # Smaller batch size to reduce memory usage
        evaluation_strategy="epoch",
        logging_dir="../logs/gpt2",
        learning_rate=5e-5,
        weight_decay=0.01,
        save_strategy="epoch",
        gradient_accumulation_steps=2,  # Simulate a larger batch size over multiple steps
        logging_first_step=True,
        logging_steps=10,  # Log every 10 steps
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train and save model
    logger.info("Starting GPT-2 training...")
    trainer.train()

    # Save the model and tokenizer
    model.save_pretrained(Config.MODEL_SAVE_PATH)
    tokenizer.save_pretrained(Config.MODEL_SAVE_PATH)
    logger.info(f"Model saved to {Config.MODEL_SAVE_PATH}")

    # Collect garbage and free memory after training
    gc.collect()

if __name__ == "__main__":
    main()
