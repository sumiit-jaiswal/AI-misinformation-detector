import numpy as np
from sklearn.metrics import classification_report
from src.utils.config import Config
from src.utils.logger import setup_logger
from transformers import BertForSequenceClassification, BertTokenizer

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://yescodersai:HPPz5fqlhvqI8KdR@cluster0.mqfoh.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

logger = setup_logger("Model Validator")

# Load test data from MongoDB
client = MongoClient(Config.MONGO_URI)
db = client[Config.DB_NAME]
test_claims = list(db.claims.find({"split": "test"}))  # Assuming test split exists

# Load model
model = BertForSequenceClassification.from_pretrained(Config.MODEL_SAVE_PATH)
tokenizer = BertTokenizer.from_pretrained(Config.MODEL_SAVE_PATH)

# Generate predictions
true_labels = []
pred_labels = []
for claim in test_claims:
    inputs = tokenizer(
        claim["text"], 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=128
    )
    outputs = model(**inputs)
    pred = np.argmax(outputs.logits.detach().numpy())
    true_labels.append(claim["label"])
    pred_labels.append(pred)

# Print report
logger.info("Classification Report:\n" + classification_report(true_labels, pred_labels))