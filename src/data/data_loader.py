# src/data/data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional
import torchvision.transforms as transforms

class GalaxyDataset(Dataset):
    """
    Dataset class for loading and preprocessing galaxy and star images.
    """
    
    def __init__(
        self,
        base_dir: str = 'data/raw',
        mode: str = 'training',
        image_size: int = 224,
        normalize: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            base_dir: Root directory containing the data
            mode: Either 'training' or 'test'
            image_size: Target size for image resizing
            normalize: Whether to normalize images
        """
        self.base_dir = Path(base_dir)
        self.mode = mode
        self.image_size = image_size
        
        # Set up paths
        self.galaxy_dir = self.base_dir / 'galaxy_postage' / mode
        self.star_dir = self.base_dir / 'star_postage' / mode
        
        # Validate directories
        if not self.galaxy_dir.exists() or not self.star_dir.exists():
            raise ValueError(f"Directory not found: {self.galaxy_dir} or {self.star_dir}")
        
        # Get file lists
        self.galaxy_paths = self._get_sorted_files('galaxy')
        self.star_paths = self._get_sorted_files('star')
        
        # Validate matching pairs
        if len(self.galaxy_paths) != len(self.star_paths):
            raise ValueError("Mismatch in number of galaxy and star images")
        
        # Set up transforms
        self.transform = self._setup_transforms(normalize)
        
        logging.info(f"Loaded {len(self.galaxy_paths)} image pairs for {mode}")

    def _get_sorted_files(self, prefix: str) -> list:
        """Get sorted list of image files with given prefix."""
        dir_path = self.galaxy_dir if prefix == 'galaxy' else self.star_dir
        return sorted([
            f for f in dir_path.glob(f'mdm_{prefix}_*.png')
            if f.is_file()
        ])

    def _setup_transforms(self, normalize: bool) -> transforms.Compose:
        """Set up image transformation pipeline."""
        transform_list = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ]
        
        if normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        
        return transforms.Compose(transform_list)

    def _load_image(self, path: Path) -> torch.Tensor:
        """Load and transform a single image."""
        try:
            img = Image.open(path).convert('RGB')
            return self.transform(img)
        except Exception as e:
            logging.error(f"Error loading image {path}: {str(e)}")
            raise

    def __len__(self) -> int:
        return len(self.galaxy_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a galaxy-star image pair.
        
        Returns:
            Dictionary containing:
                - galaxy: Transformed galaxy image tensor
                - star: Transformed star image tensor
                - galaxy_path: Path to galaxy image
                - star_path: Path to star image
        """
        try:
            galaxy_img = self._load_image(self.galaxy_paths[idx])
            star_img = self._load_image(self.star_paths[idx])
            
            return {
                'galaxy': galaxy_img,
                'star': star_img,
                'galaxy_path': str(self.galaxy_paths[idx]),
                'star_path': str(self.star_paths[idx])
            }
            
        except Exception as e:
            logging.error(f"Error loading pair at index {idx}: {str(e)}")
            raise

def create_data_loaders(
    base_dir: str = 'data/raw',
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and test data loaders.
    
    Args:
        base_dir: Root directory containing the data
        batch_size: Batch size for data loading
        num_workers: Number of worker processes for data loading
        image_size: Target size for image resizing
    
    Returns:
        Tuple of (training_loader, test_loader)
    """
    # Create datasets
    train_dataset = GalaxyDataset(
        base_dir=base_dir,
        mode='training',
        image_size=image_size
    )
    
    test_dataset = GalaxyDataset(
        base_dir=base_dir,
        mode='test',
        image_size=image_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader