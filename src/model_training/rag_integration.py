from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from pymongo import MongoClient
from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger("FAISS Builder")

def main():
    # Connect to MongoDB
    client = MongoClient(Config.MONGO_URI)
    db = client[Config.DB_NAME]
    sources = list(db.sources.find({}))
    
    # Prepare documents
    documents = [
        f"{source['title']}\n{source['text']}" 
        for source in sources
    ]

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # Build and save index
    faiss_index = FAISS.from_texts(documents, embeddings)
    faiss_index.save_local(Config.FAISS_INDEX_PATH)
    logger.info(f"FAISS index saved to {Config.FAISS_INDEX_PATH}")

if __name__ == '__main__':
    main()