from langchain_community.vectorstores import FAISS  # Updated import for FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import for HuggingFaceEmbeddings
from pymongo import MongoClient
from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger("FAISS Builder")

def main():
    # Connect to MongoDB
    client = MongoClient(Config.MONGO_URI)
    db = client[Config.DB_NAME]
    sources = list(db.sources.find({}))

    # Prepare documents with handling for missing 'text'
    documents = []
    for source in sources:
        title = source.get('title', 'No Title')
        text = source.get('text', 'No Text Available')
        
        # Log missing fields for debugging purposes
        if title == 'No Title' or text == 'No Text Available':
            logger.warning(f"Missing fields in document: {source}")
        
        documents.append(f"{title}\n{text}")
    
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
