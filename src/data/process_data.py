"""
Main Data Processing Script
Orchestrates the complete data pipeline: load → validate → preprocess → split → save
"""

import argparse
from pathlib import Path
import yaml

from src.data.data_loader import DataLoader
from src.data.validator import DataValidator
from src.data.preprocessor import DataPreprocessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataPipeline:
    """
    Complete data processing pipeline for fraud detection.
    """
    
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        """
        Initialize pipeline with configuration.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.loader = None
        self.validator = DataValidator()
        self.preprocessor = DataPreprocessor(self.config.get('preprocessing', {}))
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"✓ Configuration loaded from {config_path}")
        return config
    
    def run(
        self,
        input_path: str,
        output_dir: str = "data/processed",
        validate: bool = True,
        save_scalers: bool = True
    ) -> dict:
        """
        Run the complete data pipeline.
        
        Args:
            input_path (str): Path to raw data CSV
            output_dir (str): Directory to save processed data
            validate (bool): Whether to validate data
            save_scalers (bool): Whether to save fitted scalers
            
        Returns:
            dict: Pipeline execution results
        """
        logger.info("=" * 80)
        logger.info("STARTING DATA PIPELINE")
        logger.info("=" * 80)
        
        results = {}
        
        # Step 1: Load data
        logger.info("\n[STEP 1/5] Loading data...")
        self.loader = DataLoader(input_path)
        df = self.loader.load_data()
        results['initial_shape'] = df.shape
        
        # Step 2: Validate data
        if validate:
            logger.info("\n[STEP 2/5] Validating data...")
            is_valid, validation_results = self.validator.validate_all(df)
            results['validation'] = validation_results
            
            if not is_valid:
                logger.error("Data validation failed. Please fix errors before proceeding.")
                raise ValueError("Data validation failed")
        else:
            logger.info("\n[STEP 2/5] Skipping validation...")
        
        # Step 3: Preprocess data
        logger.info("\n[STEP 3/5] Preprocessing data...")
        
        preprocessing_config = self.config.get('preprocessing', {})
        
        df_processed = self.preprocessor.preprocess_pipeline(
            df,
            remove_duplicates=True,
            handle_missing=preprocessing_config.get('handle_missing', True),
            scale_features=preprocessing_config.get('scale_features', True),
            create_time_features=False,  # Can be enabled if needed
            fit=True
        )
        
        results['processed_shape'] = df_processed.shape
        
        # Step 4: Split data
        logger.info("\n[STEP 4/5] Splitting data...")
        
        data_config = self.config.get('data', {})
        train_split = data_config.get('train_split', 0.7)
        val_split = data_config.get('val_split', 0.15)
        test_split = data_config.get('test_split', 0.15)
        
        train_df, val_df, test_df = self.preprocessor.split_data(
            df_processed,
            train_size=train_split,
            val_size=val_split,
            test_size=test_split,
            stratify=True,
            random_state=42
        )
        
        results['split_sizes'] = {
            'train': len(train_df),
            'val': len(val_df),
            'test': len(test_df)
        }
        
        # Step 5: Save processed data
        logger.info("\n[STEP 5/5] Saving processed data...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save datasets
        train_path = output_path / 'train.csv'
        val_path = output_path / 'val.csv'
        test_path = output_path / 'test.csv'
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"✓ Train set saved: {train_path} ({len(train_df):,} rows)")
        logger.info(f"✓ Val set saved: {val_path} ({len(val_df):,} rows)")
        logger.info(f"✓ Test set saved: {test_path} ({len(test_df):,} rows)")
        
        results['output_paths'] = {
            'train': str(train_path),
            'val': str(val_path),
            'test': str(test_path)
        }
        
        # Save scalers
        if save_scalers:
            scaler_dir = output_path / 'scalers'
            self.preprocessor.save_scalers(str(scaler_dir))
            results['scaler_path'] = str(scaler_dir)
        
        # Save processing metadata
        metadata = {
            'input_path': input_path,
            'output_dir': output_dir,
            'initial_shape': results['initial_shape'],
            'processed_shape': results['processed_shape'],
            'split_sizes': results['split_sizes'],
            'preprocessing_config': preprocessing_config,
            'data_config': data_config
        }
        
        metadata_path = output_path / 'metadata.yaml'
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
        
        logger.info(f"✓ Metadata saved: {metadata_path}")
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ DATA PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"\nProcessed {results['initial_shape'][0]:,} → {results['processed_shape'][0]:,} rows")
        logger.info(f"Train: {results['split_sizes']['train']:,} | "
                   f"Val: {results['split_sizes']['val']:,} | "
                   f"Test: {results['split_sizes']['test']:,}")
        logger.info(f"\nData saved to: {output_dir}")
        logger.info("=" * 80)
        
        return results


def main():
    """Main entry point for data processing."""
    parser = argparse.ArgumentParser(description="Process credit card transaction data")
    
    parser.add_argument(
        '--input',
        type=str,
        default='data/raw/creditcard.csv',
        help='Path to raw data CSV file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed',
        help='Output directory for processed data'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/model_config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip data validation'
    )
    
    parser.add_argument(
        '--no-save-scalers',
        action='store_true',
        help='Skip saving scalers'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = DataPipeline(config_path=args.config)
    
    results = pipeline.run(
        input_path=args.input,
        output_dir=args.output,
        validate=not args.no_validate,
        save_scalers=not args.no_save_scalers
    )
    
    return results


if __name__ == "__main__":
    main()
