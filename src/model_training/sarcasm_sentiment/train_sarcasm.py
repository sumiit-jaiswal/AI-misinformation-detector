from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger("Sarcasm Trainer")

# Load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained("sismetanin/sarcasm-detection")
tokenizer = AutoTokenizer.from_pretrained("sismetanin/sarcasm-detection")

# Dummy training data (replace with actual labeled data)
train_texts = ["Oh great, another meeting!", "This is genuinely helpful"]
train_labels = [1, 0]  # 1=Sarcastic, 0=Genuine

# Tokenization
train_encodings = tokenizer(
    train_texts, truncation=True, padding=True, max_length=128
)

class SarcasmDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SarcasmDataset(train_encodings, train_labels)

# Training
training_args = TrainingArguments(
    output_dir="../results/sarcasm",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    logging_dir="../logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

logger.info("Starting sarcasm model training...")
trainer.train()
model.save_pretrained(Config.SARCASM_MODEL_PATH)
logger.info(f"Sarcasm model saved to {Config.SARCASM_MODEL_PATH}")