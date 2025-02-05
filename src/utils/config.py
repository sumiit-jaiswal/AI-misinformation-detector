import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MODEL_SAVE_PATH = "../models/finetuned_gpt2"
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    DB_NAME = "misinfo_db"
    MODEL_SAVE_PATH = "../models/finetuned_bert"
    FAISS_INDEX_PATH = "../models/faiss_index"
    SARCASM_MODEL_PATH = "../models/sarcasm_model"