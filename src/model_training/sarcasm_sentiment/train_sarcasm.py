from transformers import T5ForConditionalGeneration, AutoTokenizer, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from pymongo import MongoClient
from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger("Sarcasm Trainer")

class SarcasmDataset(Dataset):
    def __init__(self, input_encodings, target_encodings):
        self.input_ids = input_encodings['input_ids']
        self.attention_mask = input_encodings['attention_mask']
        self.labels = target_encodings['input_ids']

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

    def __len__(self):
        return len(self.input_ids)

def main():
    # Connect to MongoDB
    client = MongoClient(Config.MONGO_URI)
    db = client[Config.DB_NAME]
    sarcasm_data = list(db.sarcasm_data.find({}))
    
    # Prepare data with prefixes
    texts = ["sarcasm: " + item['text'] for item in sarcasm_data]
    label_map = {0: "not_sarcastic", 1: "sarcastic"}
    labels = [label_map[item['label']] for item in sarcasm_data]

    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained(
        "mrm8488/t5-base-finetuned-sarcasm-twitter",
        legacy=False
    )

    input_encodings = tokenizer(
        texts, 
        truncation=True, 
        padding='max_length', 
        max_length=128,
        return_tensors='pt'
    )

    target_encodings = tokenizer(
        labels,
        truncation=True,
        padding='max_length',
        max_length=8,
        return_tensors='pt'
    )

    # Split dataset
    total_samples = len(texts)
    split_idx = int(0.8 * total_samples)

    train_inputs = {k: v[:split_idx] for k, v in input_encodings.items()}
    val_inputs = {k: v[split_idx:] for k, v in input_encodings.items()}

    train_targets = {k: v[:split_idx] for k, v in target_encodings.items()}
    val_targets = {k: v[split_idx:] for k, v in target_encodings.items()}

    train_dataset = SarcasmDataset(train_inputs, train_targets)
    val_dataset = SarcasmDataset(val_inputs, val_targets)

    # Model setup
    model = T5ForConditionalGeneration.from_pretrained(
        "mrm8488/t5-base-finetuned-sarcasm-twitter"
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir='../results/sarcasm',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        evaluation_strategy='epoch',
        logging_dir='../logs/sarcasm',
        predict_with_generate=True
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