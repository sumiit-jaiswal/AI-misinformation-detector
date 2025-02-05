import json
import pandas as pd
from pathlib import Path
from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger("Data Preprocessor")

def preprocess_liar():
    raw_path = Path("data/raw/liar/train.tsv")
    df = pd.read_csv(raw_path, sep='\t', header=None)
    df = df[[1, 2]]  # Claim (col 2), Label (col 1)
    df.columns = ["label", "claim"]
    df.to_json("data/processed/liar.jsonl", orient="records", lines=True)

def preprocess_fakenewsnet():
    # Clone FakeNewsNet repo first
    claims = []
    for file in Path("data/raw/FakeNewsNet/data").rglob("*.json"):
        with open(file) as f:
            data = json.load(f)
            claims.append({
                "claim": data["text"],
                "label": 0 if data["fake"] else 1
            })
    pd.DataFrame(claims).to_json("data/processed/fakenewsnet.jsonl", orient="records", lines=True)

def main():
    preprocess_liar()
    preprocess_fakenewsnet()
    logger.info("Data preprocessing complete")

if __name__ == "__main__":
    main()
