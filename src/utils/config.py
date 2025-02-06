import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MODEL_SAVE_PATH = "../models/finetuned_gpt2"
    MONGO_URI = "mongodb+srv://yescodersai:HPPz5fqlhvqI8KdR@cluster0.mqfoh.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    DB_NAME = "misinfo_db"
    MODEL_SAVE_PATH = "../models/finetuned_bert"
    FAISS_INDEX_PATH = "../models/faiss_index"
    SARCASM_MODEL_PATH = "../models/sarcasm_model"