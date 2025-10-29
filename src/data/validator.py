"""
Data Validation Module
Ensures data quality and integrity before model training.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """
    Validates transaction data for quality and integrity issues.
    
    Performs checks on:
    - Schema compliance
    - Data types
    - Value ranges
    - Statistical properties
    - Class balance
    """
    
    def __init__(self):
        """Initialize DataValidator."""
        self.validation_results: Dict = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_schema(
        self,
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None
    ) -> bool:
        """
        Validate dataframe schema.
        
        Args:
            df (pd.DataFrame): Input dataframe
            required_columns (list, optional): List of required column names
            
        Returns:
            bool: True if schema is valid
        """
        logger.info("Validating schema...")
        
        if required_columns is None:
            # Default expected columns for credit card fraud dataset
            required_columns = ['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 29)]
        
        missing_cols = set(required_columns) - set(df.columns)
        extra_cols = set(df.columns) - set(required_columns)
        
        if missing_cols:
            error_msg = f"Missing required columns: {missing_cols}"
            self.errors.append(error_msg)
            logger.error(error_msg)
            return False
        
        if extra_cols:
            warning_msg = f"Extra columns found: {extra_cols}"
            self.warnings.append(warning_msg)
            logger.warning(warning_msg)
        
        logger.info("✓ Schema validation passed")
        return True
    
    def validate_data_types(self, df: pd.DataFrame) -> bool:
        """
        Validate column data types.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            bool: True if data types are valid
        """
        logger.info("Validating data types...")
        
        # Check for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = set(df.columns) - set(numeric_cols)
        
        # Remove categorical columns we might have added
        non_numeric_cols = non_numeric_cols - {'Time_Period'}
        
        if non_numeric_cols:
            error_msg = f"Non-numeric columns found: {non_numeric_cols}"
            self.errors.append(error_msg)
            logger.error(error_msg)
            return False
        
        # Check Class column is binary (0 or 1)
        if 'Class' in df.columns:
            unique_classes = df['Class'].unique()
            if not set(unique_classes).issubset({0, 1}):
                error_msg = f"Class column should only contain 0 or 1, found: {unique_classes}"
                self.errors.append(error_msg)
                logger.error(error_msg)
                return False
        
        logger.info("✓ Data type validation passed")
        return True
    
    def validate_missing_values(
        self,
        df: pd.DataFrame,
        max_missing_pct: float = 0.05
    ) -> bool:
        """
        Check for missing values.
        
        Args:
            df (pd.DataFrame): Input dataframe
            max_missing_pct (float): Maximum allowed percentage of missing values
            
        Returns:
            bool: True if missing values are within acceptable range
        """
        logger.info("Validating missing values...")
        
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        cols_with_missing = missing[missing > 0]
        
        if len(cols_with_missing) == 0:
            logger.info("✓ No missing values found")
            return True
        
        logger.warning(f"Found missing values in {len(cols_with_missing)} columns")
        
        critical_missing = missing_pct[missing_pct > max_missing_pct * 100]
        
        if len(critical_missing) > 0:
            error_msg = f"Columns with >{max_missing_pct*100}% missing: {critical_missing.to_dict()}"
            self.errors.append(error_msg)
            logger.error(error_msg)
            return False
        
        logger.info(f"✓ Missing values within acceptable range (<{max_missing_pct*100}%)")
        return True
    
    def validate_value_ranges(self, df: pd.DataFrame) -> bool:
        """
        Validate that values are within expected ranges.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            bool: True if all values are within valid ranges
        """
        logger.info("Validating value ranges...")
        
        valid = True
        
        # Check Time is non-negative
        if 'Time' in df.columns:
            if (df['Time'] < 0).any():
                error_msg = "Time column contains negative values"
                self.errors.append(error_msg)
                logger.error(error_msg)
                valid = False
        
        # Check Amount is non-negative
        if 'Amount' in df.columns:
            if (df['Amount'] < 0).any():
                error_msg = "Amount column contains negative values"
                self.errors.append(error_msg)
                logger.error(error_msg)
                valid = False
            
            # Check for unrealistic amounts
            max_amount = df['Amount'].max()
            if max_amount > 100000:
                warning_msg = f"Unusually high transaction amount detected: ${max_amount:,.2f}"
                self.warnings.append(warning_msg)
                logger.warning(warning_msg)
        
        # Check Class is binary
        if 'Class' in df.columns:
            if not df['Class'].isin([0, 1]).all():
                error_msg = "Class column contains values other than 0 or 1"
                self.errors.append(error_msg)
                logger.error(error_msg)
                valid = False
        
        if valid:
            logger.info("✓ Value range validation passed")
        
        return valid
    
    def validate_duplicates(
        self,
        df: pd.DataFrame,
        max_duplicate_pct: float = 0.05
    ) -> bool:
        """
        Check for duplicate rows.
        
        Args:
            df (pd.DataFrame): Input dataframe
            max_duplicate_pct (float): Maximum allowed percentage of duplicates
            
        Returns:
            bool: True if duplicates are within acceptable range
        """
        logger.info("Validating duplicates...")
        
        n_duplicates = df.duplicated().sum()
        duplicate_pct = (n_duplicates / len(df)) * 100
        
        if n_duplicates == 0:
            logger.info("✓ No duplicates found")
            return True
        
        logger.warning(f"Found {n_duplicates:,} duplicate rows ({duplicate_pct:.2f}%)")
        
        if duplicate_pct > max_duplicate_pct * 100:
            error_msg = f"Duplicate percentage ({duplicate_pct:.2f}%) exceeds threshold ({max_duplicate_pct*100}%)"
            self.errors.append(error_msg)
            logger.error(error_msg)
            return False
        
        logger.info(f"✓ Duplicates within acceptable range (<{max_duplicate_pct*100}%)")
        return True
    
    def validate_class_balance(
        self,
        df: pd.DataFrame,
        min_minority_samples: int = 50
    ) -> bool:
        """
        Validate class distribution.
        
        Args:
            df (pd.DataFrame): Input dataframe
            min_minority_samples (int): Minimum required samples in minority class
            
        Returns:
            bool: True if class balance is acceptable
        """
        logger.info("Validating class balance...")
        
        if 'Class' not in df.columns:
            warning_msg = "Class column not found, skipping class balance validation"
            self.warnings.append(warning_msg)
            logger.warning(warning_msg)
            return True
        
        class_counts = df['Class'].value_counts()
        minority_count = class_counts.min()
        majority_count = class_counts.max()
        
        imbalance_ratio = majority_count / minority_count
        fraud_rate = (class_counts.get(1, 0) / len(df)) * 100
        
        logger.info(f"Class distribution: {class_counts.to_dict()}")
        logger.info(f"Fraud rate: {fraud_rate:.4f}%")
        logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        if minority_count < min_minority_samples:
            error_msg = f"Minority class has only {minority_count} samples (minimum: {min_minority_samples})"
            self.errors.append(error_msg)
            logger.error(error_msg)
            return False
        
        if imbalance_ratio > 100:
            warning_msg = f"Severe class imbalance detected: {imbalance_ratio:.2f}:1"
            self.warnings.append(warning_msg)
            logger.warning(warning_msg)
        
        logger.info("✓ Class balance validation passed")
        return True
    
    def validate_statistical_properties(self, df: pd.DataFrame) -> bool:
        """
        Validate statistical properties of the data.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            bool: True if statistical properties are acceptable
        """
        logger.info("Validating statistical properties...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Check for columns with zero variance
        zero_var_cols = []
        for col in numeric_cols:
            if df[col].std() == 0:
                zero_var_cols.append(col)
        
        if zero_var_cols:
            warning_msg = f"Columns with zero variance: {zero_var_cols}"
            self.warnings.append(warning_msg)
            logger.warning(warning_msg)
        
        # Check for extreme outliers (beyond 5 standard deviations)
        for col in numeric_cols:
            if col in ['Time', 'Amount']:  # Skip these as they may have natural wide ranges
                continue
            
            mean = df[col].mean()
            std = df[col].std()
            
            if std > 0:
                extreme_outliers = df[np.abs((df[col] - mean) / std) > 5]
                if len(extreme_outliers) > 0:
                    warning_msg = f"{col}: {len(extreme_outliers)} extreme outliers detected (>5 std)"
                    self.warnings.append(warning_msg)
                    logger.warning(warning_msg)
        
        logger.info("✓ Statistical properties validation passed")
        return True
    
    def validate_all(
        self,
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None
    ) -> Tuple[bool, Dict]:
        """
        Run all validation checks.
        
        Args:
            df (pd.DataFrame): Input dataframe
            required_columns (list, optional): List of required columns
            
        Returns:
            Tuple[bool, Dict]: (is_valid, validation_results)
        """
        logger.info("=" * 60)
        logger.info("STARTING DATA VALIDATION")
        logger.info("=" * 60)
        
        # Reset results
        self.errors = []
        self.warnings = []
        self.validation_results = {}
        
        # Run all validations
        checks = {
            'schema': self.validate_schema(df, required_columns),
            'data_types': self.validate_data_types(df),
            'missing_values': self.validate_missing_values(df),
            'value_ranges': self.validate_value_ranges(df),
            'duplicates': self.validate_duplicates(df),
            'class_balance': self.validate_class_balance(df),
            'statistical_properties': self.validate_statistical_properties(df)
        }
        
        self.validation_results = {
            'checks': checks,
            'all_passed': all(checks.values()),
            'errors': self.errors,
            'warnings': self.warnings,
            'timestamp': datetime.now().isoformat(),
            'n_rows': len(df),
            'n_columns': len(df.columns)
        }
        
        # Summary
        logger.info("=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Checks passed: {sum(checks.values())}/{len(checks)}")
        logger.info(f"Errors: {len(self.errors)}")
        logger.info(f"Warnings: {len(self.warnings)}")
        
        if self.validation_results['all_passed']:
            logger.info("✓ ALL VALIDATIONS PASSED")
        else:
            logger.error("✗ VALIDATION FAILED")
            for error in self.errors:
                logger.error(f"  - {error}")
        
        if self.warnings:
            logger.warning("Warnings:")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")
        
        logger.info("=" * 60)
        
        return self.validation_results['all_passed'], self.validation_results


if __name__ == "__main__":
    # Example usage
    from src.data.data_loader import load_raw_data
    
    # Load data
    df = load_raw_data("data/raw/creditcard.csv")
    
    # Validate
    validator = DataValidator()
    is_valid, results = validator.validate_all(df)
    
    print(f"\nValidation {'PASSED' if is_valid else 'FAILED'}")
    print(f"Errors: {len(results['errors'])}")
    print(f"Warnings: {len(results['warnings'])}")
