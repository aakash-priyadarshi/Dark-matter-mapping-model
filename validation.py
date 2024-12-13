# validation.py
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from src.data.preprocessor import ImagePreprocessor
from src.data.data_loader import create_data_loaders
import logging
import time

class PipelineValidator:
    def __init__(self, base_dir='data/raw', batch_size=4):
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.output_dir = Path('validation_output')
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            filename=self.output_dir / 'validation.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def validate_data_loader(self):
        """Validate the data loader functionality"""
        try:
            logging.info("Starting data loader validation...")
            
            # Create data loaders
            train_loader, test_loader = create_data_loaders(
                base_dir=self.base_dir,
                batch_size=self.batch_size
            )
            
            # Check basic properties
            logging.info(f"Training samples: {len(train_loader.dataset)}")
            logging.info(f"Test samples: {len(test_loader.dataset)}")
            
            # Get a batch from each
            train_batch = next(iter(train_loader))
            test_batch = next(iter(test_loader))
            
            # Validate batch contents
            self._validate_batch(train_batch, "training")
            self._validate_batch(test_batch, "test")
            
            return True
            
        except Exception as e:
            logging.error(f"Data loader validation failed: {str(e)}")
            raise
            
    def _validate_batch(self, batch, dataset_type):
        """Validate properties of a single batch"""
        logging.info(f"\nValidating {dataset_type} batch:")
        
        # Check keys
        expected_keys = {'galaxy', 'star', 'galaxy_path', 'star_path'}
        actual_keys = set(batch.keys())
        logging.info(f"Batch keys present: {actual_keys == expected_keys}")
        
        # Check shapes
        logging.info(f"Galaxy batch shape: {batch['galaxy'].shape}")
        logging.info(f"Star batch shape: {batch['star'].shape}")
        
        # Check value ranges
        logging.info(f"Galaxy value range: [{batch['galaxy'].min():.3f}, {batch['galaxy'].max():.3f}]")
        logging.info(f"Star value range: [{batch['star'].min():.3f}, {batch['star'].max():.3f}]")
        
    def validate_preprocessor(self, batch):
        """Validate the preprocessor functionality"""
        try:
            logging.info("\nStarting preprocessor validation...")
            
            # Create preprocessor with optimized parameters
            preprocessor = ImagePreprocessor(
                gaussian_sigma=1.0,
                wavelet_sigma=0.1,
                preserve_features=True
            )
            start_time = time.time()
            
            # Process batch
            processed = preprocessor.process_batch(batch)
            
            processing_time = time.time() - start_time
            logging.info(f"Preprocessing time: {processing_time:.2f} seconds")
            
            # Validate processed output
            self._validate_processed_output(processed)
            
            return processed
            
        except Exception as e:
            logging.error(f"Preprocessor validation failed: {str(e)}")
            raise
            
    def _validate_processed_output(self, processed):
        """Validate properties of processed output"""
        logging.info("\nValidating processed output:")
        
        # Check keys
        expected_keys = {'processed_galaxies', 'processed_stars', 'paths'}
        actual_keys = set(processed.keys())
        logging.info(f"Processed keys present: {actual_keys == expected_keys}")
        
        # Check shapes
        logging.info(f"Processed galaxy shape: {processed['processed_galaxies'].shape}")
        logging.info(f"Processed star shape: {processed['processed_stars'].shape}")
        
        # Check value ranges
        logging.info(f"Processed galaxy range: [{processed['processed_galaxies'].min():.3f}, "
                    f"{processed['processed_galaxies'].max():.3f}]")
        logging.info(f"Processed star range: [{processed['processed_stars'].min():.3f}, "
                    f"{processed['processed_stars'].max():.3f}]")
        
    def visualize_results(self, original_batch, processed_batch):
        """Create visualizations of original and processed images"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Plot original and processed images
        for i in range(min(2, self.batch_size)):
            # Original galaxy - denormalize for visualization
            galaxy_img = original_batch['galaxy'][i].cpu()
            star_img = original_batch['star'][i].cpu()
            
            # Convert to grayscale for better visualization
            galaxy_img = 0.299 * galaxy_img[0] + 0.587 * galaxy_img[1] + 0.114 * galaxy_img[2]
            star_img = 0.299 * star_img[0] + 0.587 * star_img[1] + 0.114 * star_img[2]
            
            # Enhance contrast for visualization
            def enhance_contrast(img):
                p2, p98 = torch.quantile(img.flatten(), torch.tensor([0.02, 0.98]))
                return torch.clamp((img - p2) / (p98 - p2), 0, 1)
            
            galaxy_img = enhance_contrast(galaxy_img)
            star_img = enhance_contrast(star_img)
            
            axes[0,i*2].imshow(galaxy_img, cmap='gray')
            axes[0,i*2].set_title(f'Original Galaxy {i+1}')
            
            axes[0,i*2+1].imshow(star_img, cmap='gray')
            axes[0,i*2+1].set_title(f'Original Star {i+1}')
            
            # Processed images
            axes[1,i*2].imshow(processed_batch['processed_galaxies'][i].cpu(), cmap='gray')
            axes[1,i*2].set_title(f'Processed Galaxy {i+1}')
            
            axes[1,i*2+1].imshow(processed_batch['processed_stars'][i].cpu(), cmap='gray')
            axes[1,i*2+1].set_title(f'Processed Star {i+1}')
        
        # Turn off axes
        for ax in axes.flat:
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pipeline_validation.png')
        plt.close()
        
    def run_full_validation(self):
        """Run complete validation of the pipeline"""
        try:
            logging.info("Starting full pipeline validation...")
            
            # Validate data loader
            self.validate_data_loader()
            
            # Get a batch for preprocessing
            train_loader, _ = create_data_loaders(
                base_dir=self.base_dir,
                batch_size=self.batch_size
            )
            batch = next(iter(train_loader))
            
            # Validate preprocessor
            processed_batch = self.validate_preprocessor(batch)
            
            # Create visualizations
            self.visualize_results(batch, processed_batch)
            
            logging.info("Pipeline validation completed successfully!")
            print("Validation completed! Check validation_output/ for results.")
            return True
            
        except Exception as e:
            logging.error(f"Pipeline validation failed: {str(e)}")
            print(f"Validation failed! Check validation_output/validation.log for details.")
            raise

if __name__ == "__main__":
    validator = PipelineValidator()
    validator.run_full_validation()