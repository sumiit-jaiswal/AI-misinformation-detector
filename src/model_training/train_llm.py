import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from src.utils.config import Config
from src.utils.logger import setup_logger
import argparse

logger = setup_logger("LLM Trainer")

class LLMTrainer:
    def __init__(self, model_type="bert"):
        self.model_type = model_type
        self.tokenizer = None
        self.model = None
        
        # Model configurations
        self.config = {
            "bert": {
                "model_name": "bert-base-uncased",
                "padding": "max_length",
                "max_length": 128,
                "num_labels": 3
            },
            "gpt2": {
                "model_name": "gpt2",
                "padding": "max_length",
                "max_length": 128,
                "num_labels": 2,
                "special_tokens": {"pad_token": "<PAD>"}
            }
        }
        
        self._load_model()
        
    def _load_model(self):
        """Initialize model and tokenizer based on type"""
        cfg = self.config[self.model_type]
        
        # Special token handling for GPT-2
        if self.model_type == "gpt2":
            self.tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
            self.tokenizer.add_special_tokens(cfg["special_tokens"])
            self.model = AutoModelForSequenceClassification.from_pretrained(
                cfg["model_name"],
                num_labels=cfg["num_labels"],
                pad_token_id=self.tokenizer.pad_token_id
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
            self.model = AutoModelForSequenceClassification.from_pretrained(
                cfg["model_name"],
                num_labels=cfg["num_labels"]
            )

    def _prepare_data(self):
        """Load and process data from MongoDB"""
        client = MongoClient(Config.MONGO_URI)
        db = client[Config.DB_NAME]
        
        # Different data handling for GPT-2 (binary classification)
        if self.model_type == "gpt2":
            claims = list(db.gpt2_claims.find({}))
            texts = [c["text"] for c in claims]
            labels = [1 if c["label"] in ["true", "real"] else 0 for c in claims]
        else:  # BERT
            claims = list(db.claims.find({}))
            texts = [c["text"] for c in claims]
            labels = [c["label"] for c in claims]
            
        return train_test_split(texts, labels, test_size=0.2, random_state=42)

    def _tokenize(self, texts):
        """Generic tokenization handling"""
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=self.config[self.model_type]["padding"],
            max_length=self.config[self.model_type]["max_length"],
            return_tensors='pt'
        )
        
        # Handle GPT-2's special token padding (if model is GPT-2)
        if self.model_type == "gpt2":
            encodings["attention_mask"] = torch.ones_like(encodings["input_ids"])  # GPT-2 doesn't have attention_mask by default
        
        return encodings

    def train(self):
        """Main training workflow"""
        train_texts, val_texts, train_labels, val_labels = self._prepare_data()
        
        # Tokenization
        train_encodings = self._tokenize(train_texts)
        val_encodings = self._tokenize(val_texts)

        # Dataset class
        class MisinfoDataset(torch.utils.data.Dataset):
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

        # Create datasets
        train_dataset = MisinfoDataset(train_encodings, train_labels)
        val_dataset = MisinfoDataset(val_encodings, val_labels)

        # Training setup
        training_args = TrainingArguments(
            output_dir=f"../results/{self.model_type}",
            num_train_epochs=3,
            per_device_train_batch_size=8 if self.model_type == "gpt2" else 16,
            evaluation_strategy="epoch",  # updated from deprecated 'eval_strategy'
            learning_rate=5e-5,
            logging_dir=f"../logs/{self.model_type}",
            save_strategy="epoch"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        # Train and save
        logger.info(f"Starting {self.model_type.upper()} training...")
        trainer.train()
        
        save_path = f"../models/finetuned_{self.model_type}"
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["bert", "gpt2"], required=True)
    args = parser.parse_args()
    
    trainer = LLMTrainer(model_type=args.model_type)
    trainer.train()
