"""
Data Preprocessing Module
Handles cleaning, transformation, and feature scaling of transaction data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """
    Preprocesses transaction data for fraud detection modeling.
    
    Handles:
    - Duplicate removal
    - Feature scaling
    - Train/validation/test splitting
    - Data persistence
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config (dict, optional): Configuration dictionary with preprocessing parameters
        """
        self.config = config or {}
        
        # Scalers
        self.amount_scaler: Optional[StandardScaler] = None
        self.time_scaler: Optional[StandardScaler] = None
        
        # Track preprocessing steps
        self.is_fitted = False
        self.feature_names: Optional[list] = None
        
    def remove_duplicates(self, df: pd.DataFrame, keep: str = 'first') -> pd.DataFrame:
        """
        Remove duplicate rows from dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            keep (str): Which duplicate to keep ('first', 'last', False)
            
        Returns:
            pd.DataFrame: Dataframe with duplicates removed
        """
        n_duplicates = df.duplicated().sum()
        
        if n_duplicates > 0:
            logger.info(f"Removing {n_duplicates:,} duplicate rows (keep='{keep}')")
            df_clean = df.drop_duplicates(keep=keep).reset_index(drop=True)
            logger.info(f"✓ Duplicates removed. Rows: {len(df):,} → {len(df_clean):,}")
            return df_clean
        else:
            logger.info("No duplicates found")
            return df.copy()
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
        """
        Handle missing values in the dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            strategy (str): Strategy for handling missing values ('drop', 'mean', 'median', 'mode')
            
        Returns:
            pd.DataFrame: Dataframe with missing values handled
        """
        missing_count = df.isnull().sum().sum()
        
        if missing_count == 0:
            logger.info("No missing values found")
            return df.copy()
        
        logger.info(f"Found {missing_count:,} missing values")
        
        if strategy == 'drop':
            df_clean = df.dropna().reset_index(drop=True)
            logger.info(f"✓ Dropped rows with missing values. Rows: {len(df):,} → {len(df_clean):,}")
            
        elif strategy == 'mean':
            df_clean = df.fillna(df.mean(numeric_only=True))
            logger.info("✓ Filled missing values with mean")
            
        elif strategy == 'median':
            df_clean = df.fillna(df.median(numeric_only=True))
            logger.info("✓ Filled missing values with median")
            
        elif strategy == 'mode':
            df_clean = df.fillna(df.mode().iloc[0])
            logger.info("✓ Filled missing values with mode")
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return df_clean
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale Amount and Time features using StandardScaler.
        
        Args:
            df (pd.DataFrame): Input dataframe
            fit (bool): Whether to fit the scaler (True for training, False for test/validation)
            
        Returns:
            pd.DataFrame: Dataframe with scaled features
        """
        df_scaled = df.copy()
        
        # Scale Amount
        if 'Amount' in df_scaled.columns:
            if fit or self.amount_scaler is None:
                logger.info("Fitting scaler for Amount feature")
                self.amount_scaler = StandardScaler()
                df_scaled['Amount'] = self.amount_scaler.fit_transform(
                    df_scaled[['Amount']]
                )
            else:
                df_scaled['Amount'] = self.amount_scaler.transform(
                    df_scaled[['Amount']]
                )
            logger.info("✓ Amount feature scaled")
        
        # Scale Time
        if 'Time' in df_scaled.columns:
            if fit or self.time_scaler is None:
                logger.info("Fitting scaler for Time feature")
                self.time_scaler = StandardScaler()
                df_scaled['Time'] = self.time_scaler.fit_transform(
                    df_scaled[['Time']]
                )
            else:
                df_scaled['Time'] = self.time_scaler.transform(
                    df_scaled[['Time']]
                )
            logger.info("✓ Time feature scaled")
        
        self.is_fitted = True
        return df_scaled
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional time-based features.
        
        Args:
            df (pd.DataFrame): Input dataframe with 'Time' column
            
        Returns:
            pd.DataFrame: Dataframe with additional time features
        """
        df_enhanced = df.copy()
        
        if 'Time' not in df_enhanced.columns:
            logger.warning("'Time' column not found, skipping time feature creation")
            return df_enhanced
        
        logger.info("Creating time-based features")
        
        # Convert seconds to hours
        df_enhanced['Time_Hour'] = df_enhanced['Time'] / 3600
        
        # Hour of day (0-23) - assuming Time is seconds from start
        df_enhanced['Hour_Of_Day'] = (df_enhanced['Time'] / 3600) % 24
        
        # Time period (Night: 0-6, Morning: 6-12, Afternoon: 12-18, Evening: 18-24)
        df_enhanced['Time_Period'] = pd.cut(
            df_enhanced['Hour_Of_Day'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            include_lowest=True
        )
        
        logger.info("✓ Time features created: Time_Hour, Hour_Of_Day, Time_Period")
        
        return df_enhanced
    
    def preprocess_pipeline(
        self,
        df: pd.DataFrame,
        remove_duplicates: bool = True,
        handle_missing: bool = True,
        scale_features: bool = True,
        create_time_features: bool = False,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            df (pd.DataFrame): Input dataframe
            remove_duplicates (bool): Whether to remove duplicates
            handle_missing (bool): Whether to handle missing values
            scale_features (bool): Whether to scale Amount and Time
            create_time_features (bool): Whether to create additional time features
            fit (bool): Whether to fit scalers
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        logger.info("=" * 60)
        logger.info("STARTING PREPROCESSING PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Initial shape: {df.shape}")
        
        df_processed = df.copy()
        
        # Step 1: Remove duplicates
        if remove_duplicates:
            df_processed = self.remove_duplicates(df_processed)
        
        # Step 2: Handle missing values
        if handle_missing:
            df_processed = self.handle_missing_values(df_processed, strategy='drop')
        
        # Step 3: Create time features (before scaling)
        if create_time_features:
            df_processed = self.create_time_features(df_processed)
        
        # Step 4: Scale features
        if scale_features:
            df_processed = self.scale_features(df_processed, fit=fit)
        
        logger.info(f"Final shape: {df_processed.shape}")
        logger.info("=" * 60)
        logger.info("✓ PREPROCESSING COMPLETE")
        logger.info("=" * 60)
        
        return df_processed
    
    def split_data(
        self,
        df: pd.DataFrame,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        stratify: bool = True,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df (pd.DataFrame): Input dataframe
            train_size (float): Proportion for training set
            val_size (float): Proportion for validation set
            test_size (float): Proportion for test set
            stratify (bool): Whether to stratify by Class
            random_state (int): Random seed
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train, val, test dataframes
        """
        if not np.isclose(train_size + val_size + test_size, 1.0):
            raise ValueError(f"Split sizes must sum to 1.0, got {train_size + val_size + test_size}")
        
        logger.info("=" * 60)
        logger.info("SPLITTING DATA")
        logger.info("=" * 60)
        
        stratify_column = df['Class'] if stratify and 'Class' in df.columns else None
        
        # First split: train + val vs test
        train_val_size = train_size + val_size
        train_val, test = train_test_split(
            df,
            test_size=test_size,
            stratify=stratify_column,
            random_state=random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / train_val_size
        stratify_column_tv = train_val['Class'] if stratify and 'Class' in train_val.columns else None
        
        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            stratify=stratify_column_tv,
            random_state=random_state
        )
        
        logger.info(f"Train set: {len(train):,} samples ({train_size*100:.1f}%)")
        logger.info(f"Val set:   {len(val):,} samples ({val_size*100:.1f}%)")
        logger.info(f"Test set:  {len(test):,} samples ({test_size*100:.1f}%)")
        
        if 'Class' in df.columns:
            logger.info("\nClass distribution:")
            logger.info(f"Train - Fraud: {train['Class'].sum():,} ({train['Class'].mean()*100:.4f}%)")
            logger.info(f"Val   - Fraud: {val['Class'].sum():,} ({val['Class'].mean()*100:.4f}%)")
            logger.info(f"Test  - Fraud: {test['Class'].sum():,} ({test['Class'].mean()*100:.4f}%)")
        
        logger.info("=" * 60)
        
        return train, val, test
    
    def save_scalers(self, save_dir: str) -> None:
        """
        Save fitted scalers to disk.
        
        Args:
            save_dir (str): Directory to save scalers
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self.amount_scaler is not None:
            amount_path = save_path / 'amount_scaler.pkl'
            joblib.dump(self.amount_scaler, amount_path)
            logger.info(f"✓ Amount scaler saved to {amount_path}")
        
        if self.time_scaler is not None:
            time_path = save_path / 'time_scaler.pkl'
            joblib.dump(self.time_scaler, time_path)
            logger.info(f"✓ Time scaler saved to {time_path}")
    
    def load_scalers(self, load_dir: str) -> None:
        """
        Load fitted scalers from disk.
        
        Args:
            load_dir (str): Directory containing saved scalers
        """
        load_path = Path(load_dir)
        
        amount_path = load_path / 'amount_scaler.pkl'
        if amount_path.exists():
            self.amount_scaler = joblib.load(amount_path)
            logger.info(f"✓ Amount scaler loaded from {amount_path}")
        
        time_path = load_path / 'time_scaler.pkl'
        if time_path.exists():
            self.time_scaler = joblib.load(time_path)
            logger.info(f"✓ Time scaler loaded from {time_path}")
        
        self.is_fitted = True


if __name__ == "__main__":
    # Example usage
    from src.data.data_loader import load_raw_data
    
    # Load data
    df = load_raw_data("data/raw/creditcard.csv")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Preprocess
    df_processed = preprocessor.preprocess_pipeline(df, create_time_features=True)
    
    # Split data
    train, val, test = preprocessor.split_data(df_processed)
    
    print("\nProcessing complete!")
    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
