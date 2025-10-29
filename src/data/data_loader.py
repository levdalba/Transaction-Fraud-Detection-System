"""
Data Loader Module
Handles loading raw transaction data with validation and error handling.
"""

import pandas as pd
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """
    Loads and performs initial validation on raw transaction data.
    
    Attributes:
        data_path (Path): Path to the raw data file
        df (pd.DataFrame): Loaded dataframe
    """
    
    def __init__(self, data_path: str):
        """
        Initialize DataLoader.
        
        Args:
            data_path (str): Path to the CSV file
        """
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file with error handling.
        
        Returns:
            pd.DataFrame: Loaded dataframe
            
        Raises:
            ValueError: If data cannot be loaded or is invalid
        """
        try:
            logger.info(f"Loading data from {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            
            # Basic validation
            if self.df.empty:
                raise ValueError("Loaded dataframe is empty")
            
            logger.info(f"✓ Data loaded successfully: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
            
            # Log data info
            self._log_data_info()
            
            return self.df
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def _log_data_info(self) -> None:
        """Log basic information about the loaded data."""
        if self.df is None:
            return
        
        logger.info("=" * 60)
        logger.info("DATA SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Shape: {self.df.shape}")
        logger.info(f"Columns: {list(self.df.columns)}")
        logger.info(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Check for missing values
        missing = self.df.isnull().sum().sum()
        logger.info(f"Missing values: {missing}")
        
        # Check for duplicates
        duplicates = self.df.duplicated().sum()
        logger.info(f"Duplicate rows: {duplicates}")
        
        # Class distribution if available
        if 'Class' in self.df.columns:
            class_dist = self.df['Class'].value_counts()
            logger.info(f"Class distribution:\n{class_dist}")
            fraud_rate = (class_dist.get(1, 0) / len(self.df)) * 100
            logger.info(f"Fraud rate: {fraud_rate:.4f}%")
        
        logger.info("=" * 60)
    
    def get_data_summary(self) -> dict:
        """
        Get summary statistics of the loaded data.
        
        Returns:
            dict: Summary statistics
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        summary = {
            'n_rows': len(self.df),
            'n_columns': len(self.df.columns),
            'columns': list(self.df.columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicates': self.df.duplicated().sum(),
            'memory_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'dtypes': self.df.dtypes.to_dict()
        }
        
        if 'Class' in self.df.columns:
            summary['class_distribution'] = self.df['Class'].value_counts().to_dict()
            summary['fraud_rate'] = (self.df['Class'].sum() / len(self.df)) * 100
        
        return summary
    
    def validate_schema(self, expected_columns: list) -> bool:
        """
        Validate that the dataframe has expected columns.
        
        Args:
            expected_columns (list): List of expected column names
            
        Returns:
            bool: True if schema is valid
            
        Raises:
            ValueError: If schema validation fails
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        missing_cols = set(expected_columns) - set(self.df.columns)
        extra_cols = set(self.df.columns) - set(expected_columns)
        
        if missing_cols:
            raise ValueError(f"Missing expected columns: {missing_cols}")
        
        if extra_cols:
            logger.warning(f"Extra columns found: {extra_cols}")
        
        logger.info("✓ Schema validation passed")
        return True


def load_raw_data(data_path: str) -> pd.DataFrame:
    """
    Convenience function to load raw data.
    
    Args:
        data_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    loader = DataLoader(data_path)
    return loader.load_data()


if __name__ == "__main__":
    # Example usage
    data_path = "data/raw/creditcard.csv"
    loader = DataLoader(data_path)
    df = loader.load_data()
    print(f"\nLoaded {len(df):,} transactions")
    
    # Get summary
    summary = loader.get_data_summary()
    print(f"Fraud rate: {summary['fraud_rate']:.4f}%")
