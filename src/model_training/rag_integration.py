from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from src.utils.config import Config
from src.utils.logger import setup_logger
from pymongo import MongoClient

logger = setup_logger("RAG Setup")

# Initialize MongoDB connection
client = MongoClient(Config.MONGO_URI)
db = client[Config.DB_NAME]
sources = list(db.sources.find({}))

# Prepare documents
documents = [
    f"{source['title']} {source['text']}" 
    for source in sources
]

# Create FAISS index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
faiss_index = FAISS.from_texts(documents, embeddings)

# Save index
faiss_index.save_local(Config.FAISS_INDEX_PATH)
logger.info(f"FAISS index saved to {Config.FAISS_INDEX_PATH}")