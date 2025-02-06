# main.py
from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import redis

load_dotenv()
app = FastAPI()

# MongoDB Connection
mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client["misinfo_db"]

# Redis Caching
redis_client = redis.Redis(host="localhost", port=6379, db=0)

@app.post("/verify")
async def verify_claim(claim: str):
    # Check cache first
    cached_result = redis_client.get(claim)
    if cached_result:
        return {"result": cached_result.decode()}

    # Process claim using LLM + RAG (mock function)
    result = process_claim(claim)  # Replace with your model inference logic

    # Update cache and database
    redis_client.setex(claim, 3600, result)  # Cache for 1 hour
    db.claims.insert_one({"claim": claim, "result": result})
    
    return {"result": result}