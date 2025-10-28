"""
Kaggle Dataset Downloader
Download the Credit Card Fraud Detection dataset from Kaggle
"""

import os
import zipfile
from pathlib import Path

def setup_kaggle_credentials():
    """Check and setup Kaggle credentials"""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if not kaggle_json.exists():
        print("Kaggle credentials not found!")
        print("\nPlease follow these steps:")
        print("1. Go to https://www.kaggle.com/settings/account")
        print("2. Scroll down to 'API' section")
        print("3. Click 'Create New Token'")
        print("4. This will download 'kaggle.json'")
        print(f"5. Move it to: {kaggle_dir}/")
        print(f"\nRun: mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/")
        print("       chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    return True

def download_dataset():
    """Download Credit Card Fraud Detection dataset"""
    if not setup_kaggle_credentials():
        return
    
    import kaggle
    
    data_dir = Path(__file__).parent.parent / 'data' / 'raw'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading Credit Card Fraud Detection dataset...")
    print("Dataset: mlg-ulb/creditcardfraud")
    
    kaggle.api.dataset_download_files(
        'mlg-ulb/creditcardfraud',
        path=str(data_dir),
        unzip=True
    )
    
    print(f"\nDataset downloaded successfully to: {data_dir}")
    
    csv_file = data_dir / 'creditcard.csv'
    if csv_file.exists():
        file_size = csv_file.stat().st_size / (1024 * 1024)
        print(f"File: creditcard.csv ({file_size:.2f} MB)")
        
        import pandas as pd
        df = pd.read_csv(csv_file, nrows=5)
        print(f"\nDataset shape (first 5 rows): {df.shape}")
        print("\nColumns:")
        print(df.columns.tolist())

if __name__ == "__main__":
    download_dataset()
