import json
import pandas as pd
from pathlib import Path
from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger("Data Preprocessor")

# Preprocess the LIAR dataset
def preprocess_liar():
    """
    Preprocess the LIAR dataset by reading the train.tsv file,
    extracting claims and labels, and saving them as JSON Lines.
    """
    raw_path = Path("data/raw/liar/train.tsv")
    
    try:
        # Read the LIAR dataset file
        df = pd.read_csv(raw_path, sep='\t', header=None)
        # Check the first few rows to confirm the structure
        logger.info(f"Loaded LIAR dataset with {len(df)} rows.")
        
        # Extract claim (col 2) and label (col 1) from the dataset
        df = df[[1, 2]]
        df.columns = ["label", "claim"]
        
        # Ensure labels are valid
        df['label'] = df['label'].apply(lambda x: 1 if x in ["true", "real"] else 0)

        # Save the processed dataset as a JSONL file
        df.to_json("data/processed/liar.jsonl", orient="records", lines=True)
        logger.info("LIAR dataset preprocessing complete.")
    except Exception as e:
        logger.error(f"Error processing LIAR dataset: {e}")

# Preprocess the FakeNewsNet dataset
def preprocess_fakenewsnet():
    """
    Preprocess the FakeNewsNet dataset by reading CSV files containing claims and labels,
    then saving them as JSON Lines.
    """
    claims = []
    try:
        data_folder = Path("data/raw/FakeNewsNet/data")
        logger.info(f"Searching for CSV files in {data_folder}")
        
        # List all relevant CSV files (politifact_fake.csv, politifact_real.csv, gossipcop_fake.csv, gossipcop_real.csv)
        csv_files = list(data_folder.rglob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files.")
        
        # Iterate over each CSV file
        for file in csv_files:
            try:
                # Read the CSV file
                df = pd.read_csv(file)
                logger.info(f"Loaded {file} with {len(df)} rows.")
                
                # Extract claims and labels from the CSV file
                for _, row in df.iterrows():
                    # Extract the label (0 for fake, 1 for real)
                    label = 0 if 'fake' in file.name.lower() else 1
                    # Extract the claim (title of the news article)
                    claims.append({
                        "claim": row["title"],  # The 'title' column contains the news title/claim
                        "label": label
                    })
                
            except Exception as e:
                logger.error(f"Error processing file {file}: {e}")
                continue

        # Check how many claims we have
        logger.info(f"Loaded {len(claims)} claims from FakeNewsNet dataset.")
        
        if claims:  # Only save if there are claims
            # Save the claims and labels into a JSONL file
            pd.DataFrame(claims).to_json("data/processed/fakenewsnet.jsonl", orient="records", lines=True)
            logger.info("FakeNewsNet dataset preprocessing complete.")
        else:
            logger.warning("No claims found in the FakeNewsNet dataset.")

    except Exception as e:
        logger.error(f"Error processing FakeNewsNet dataset: {e}")

# Main function to execute the preprocessing
def main():
    logger.info("Starting data preprocessing...")

    # Preprocess the LIAR and FakeNewsNet datasets
    preprocess_liar()
    preprocess_fakenewsnet()

    # Final log
    logger.info("Data preprocessing complete.")
    
if __name__ == "__main__":
    main()
