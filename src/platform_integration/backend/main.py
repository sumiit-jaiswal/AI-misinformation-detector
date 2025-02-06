# main.py
import os
from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from dotenv import load_dotenv
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import json

# Load environment variables
load_dotenv()

# FastAPI application setup
app = FastAPI()

# MongoDB Connection
mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client["misinfo_db"]

# Directory paths from environment variables
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH")

# Load GPT-2 model and tokenizer
def load_gpt2_model():
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_SAVE_PATH)
    model = GPT2ForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
    return tokenizer, model

# Retrieve relevant documents using FAISS
def retrieve_information(query: str) -> str:
    # Load FAISS index and embeddings
    faiss_index = FAISS.load_local(FAISS_INDEX_PATH, HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
    
    # Retrieve relevant documents for the query
    retrieved_docs = faiss_index.similarity_search(query, k=3)  # Retrieve top 3 most relevant documents
    retrieved_text = "\n".join([doc['text'] for doc in retrieved_docs])  # Concatenate document texts
    
    return retrieved_text

# Process claim using RAG with GPT-2
def process_claim_with_rag(claim: str) -> str:
    # Step 1: Retrieve relevant documents from the FAISS index
    retrieved_info = retrieve_information(claim)
    
    # Step 2: Combine the retrieved information with the claim for processing
    full_input = f"Claim: {claim}\n\nContext:\n{retrieved_info}"
    
    # Step 3: Load the GPT-2 model and tokenizer
    tokenizer, model = load_gpt2_model()
    
    # Step 4: Tokenize the combined input (claim + retrieved information)
    inputs = tokenizer(full_input, return_tensors='pt', truncation=True, padding=True, max_length=512)
    
    # Step 5: Predict class label (True, False, or Uncertain)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

    # Step 6: Map the predicted class to a label
    if predicted_class == 0:
        return "False"
    elif predicted_class == 1:
        return "True"
    else:
        return "Uncertain"

# API route to verify claim
@app.post("/verify")
async def verify_claim(claim: str):
    # Process claim using RAG
    result = process_claim_with_rag(claim)

    # Store the result in MongoDB
    db.claims.insert_one({"claim": claim, "result": result})
    
    return {"result": result}

# Build FAISS index from MongoDB documents
def build_faiss_index():
    # Connect to MongoDB
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client[os.getenv("DB_NAME")]
    sources = list(db.sources.find({}))
    
    # Prepare documents with handling for missing 'text'
    documents = []
    for source in sources:
        title = source.get('title', 'No Title')
        text = source.get('text', 'No Text Available')
        documents.append(f"{title}\n{text}")
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # Build and save the FAISS index
    faiss_index = FAISS.from_texts(documents, embeddings)
    faiss_index.save_local(FAISS_INDEX_PATH)
    print(f"FAISS index saved to {FAISS_INDEX_PATH}")

# Check if FAISS index exists, if not, build it
if not os.path.exists(FAISS_INDEX_PATH):
    print("FAISS index not found. Building index...")
    build_faiss_index()

# Run FastAPI server: `uvicorn main:app --reload`
