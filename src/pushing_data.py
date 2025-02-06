# mongo_handler.py
import logging
from pymongo import MongoClient, errors
from dotenv import load_dotenv
import os
import uuid
from typing import Optional, Dict, Any
from datetime import datetime
from pymongo.server_api import ServerApi

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MongoDBHandler")

load_dotenv()

class MongoDBHandler:
    def __init__(self):
        self.uri = os.getenv("MONGO_URI")
        if not self.uri:
            raise ValueError("MONGO_URI environment variable not set")
        
        self.client = None
        self.db = None
        self.claims = None
        self.sources = None

    def __enter__(self):
        """Context manager entry point"""
        self._connect()
        self._initialize_collections()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point"""
        self.close()

    def _connect(self):
        """Establish database connection with error handling"""
        try:
            self.client = MongoClient(
                self.uri,
                server_api=ServerApi('1'),
                connectTimeoutMS=5000,
                serverSelectionTimeoutMS=5000
            )
            self.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")
        except errors.ConnectionFailure as e:
            logger.error(f"Connection failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected connection error: {str(e)}")
            raise

    def _initialize_collections(self):
        """Initialize database and collections with validation"""
        try:
            self.db = self.client["misinfo_db"]
            
            # Create collections with schema validation
            self.claims = self.db.create_collection("claims", validator={
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["text", "platform", "timestamp"],
                    "properties": {
                        "claim_id": {"bsonType": "string"},
                        "text": {"bsonType": "string"},
                        "platform": {"bsonType": "string"},
                        "timestamp": {"bsonType": "date"},
                    }
                }
            })
            
            self.sources = self.db.create_collection("sources", validator={
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["title", "url", "publisher"],
                    "properties": {
                        "source_id": {"bsonType": "string"},
                        "title": {"bsonType": "string"},
                        "url": {"bsonType": "string"},
                        "publisher": {"bsonType": "string"},
                        "timestamp": {"bsonType": "date"},
                    }
                }
            })
            
            # Create indexes
            self.claims.create_index([("text", "text")])
            self.sources.create_index([("url", 1)], unique=True)
            
        except errors.CollectionInvalid as e:
            logger.warning(f"Collection already exists: {str(e)}")
            self.claims = self.db.claims
            self.sources = self.db.sources
        except Exception as e:
            logger.error(f"Collection initialization failed: {str(e)}")
            raise

    def insert_claim(self, claim_data: Dict[str, Any]) -> Optional[str]:
        """Insert a claim document with validation"""
        try:
            # Validate required fields
            if not all(k in claim_data for k in ["text", "platform"]):
                raise ValueError("Missing required fields: text or platform")
                
            # Generate document ID and timestamp
            claim_data.setdefault("claim_id", str(uuid.uuid4()))
            claim_data.setdefault("timestamp", datetime.utcnow())
            
            result = self.claims.insert_one(claim_data)
            logger.info(f"Inserted claim ID: {result.inserted_id}")
            return str(result.inserted_id)
            
        except errors.WriteError as e:
            logger.error(f"Validation error: {str(e)}")
        except errors.DuplicateKeyError as e:
            logger.error(f"Duplicate claim ID: {str(e)}")
        except Exception as e:
            logger.error(f"Error inserting claim: {str(e)}")
        return None

    def insert_source(self, source_data: Dict[str, Any]) -> Optional[str]:
        """Insert a source document with validation"""
        try:
            # Validate required fields
            if not all(k in source_data for k in ["title", "url", "publisher"]):
                raise ValueError("Missing required fields: title, url, or publisher")
                
            # Generate document ID and timestamp
            source_data.setdefault("source_id", str(uuid.uuid4()))
            source_data.setdefault("timestamp", datetime.utcnow())
            
            result = self.sources.insert_one(source_data)
            logger.info(f"Inserted source ID: {result.inserted_id}")
            return str(result.inserted_id)
            
        except errors.WriteError as e:
            logger.error(f"Validation error: {str(e)}")
        except errors.DuplicateKeyError as e:
            logger.error(f"Duplicate source URL: {str(e)}")
        except Exception as e:
            logger.error(f"Error inserting source: {str(e)}")
        return None

    def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

# Modified example usage section
if __name__ == "__main__":
    try:
        with MongoDBHandler() as handler:
            # Case 1: False claim (label 0)
            false_claim = {
                "text": "COVID-19 vaccines contain microchips",
                "platform": "Facebook",
                "label": 0,
                "author": "user789",
                "language": "English"
            }
            false_id = handler.insert_claim(false_claim)
            logger.info(f"Inserted FALSE claim: {false_id}")

            # Case 2: True claim (label 1)
            true_claim = {
                "text": "WHO recommends regular handwashing to prevent COVID-19",
                "platform": "WHO Official Site",
                "label": 1,
                "source_url": "https://www.who.int",
                "language": "English"
            }
            true_id = handler.insert_claim(true_claim)
            logger.info(f"Inserted TRUE claim: {true_id}")

            # Case 3: Ambiguous claim (label 2)
            ambiguous_claim = {
                "text": "5G technology may have unknown health effects",
                "platform": "Twitter",
                "label": 2,
                "context": "Under scientific investigation",
                "language": "English"
            }
            ambiguous_id = handler.insert_claim(ambiguous_claim)
            logger.info(f"Inserted AMBIGUOUS claim: {ambiguous_id}")

            # Insert a reliable source
            reliable_source = {
                "title": "CDC COVID-19 Fact Sheet",
                "url": "https://www.cdc.gov/coronavirus/2019-ncov/index.html",
                "publisher": "Centers for Disease Control and Prevention",
                "reliability_score": 9.8
            }
            source_id = handler.insert_source(reliable_source)
            logger.info(f"Inserted reliable source: {source_id}")

    except Exception as e:
        logger.error(f"Application error: {str(e)}")