# src/data/preprocessor.py
import torch
import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d
import cv2
from skimage.restoration import denoise_wavelet
from skimage.filters import gaussian

class ImagePreprocessor:
    def __init__(self, 
                 uncertainty_threshold=0.05,
                 gaussian_sigma=1.0,
                 wavelet_sigma=0.1,
                 preserve_features=True):
        """
        Enhanced image preprocessor with better feature preservation.
        
        Args:
            uncertainty_threshold (float): Uncertainty in total light (~5%)
            gaussian_sigma (float): Sigma for Gaussian denoising
            wavelet_sigma (float): Sigma for wavelet denoising
            preserve_features (bool): Whether to preserve fine features
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.uncertainty_threshold = uncertainty_threshold
        self.gaussian_sigma = gaussian_sigma
        self.wavelet_sigma = wavelet_sigma
        self.preserve_features = preserve_features

    def remove_background(self, image):
        """Enhanced background subtraction with edge preservation"""
        # Estimate background using a large kernel
        background = ndimage.gaussian_filter(image, sigma=10.0)
        # Subtract background while preserving edges
        cleaned = image - background
        return np.clip(cleaned, 0, None)  # Ensure non-negative values

    def denoise_image(self, image):
        """Multi-scale denoising with feature preservation"""
        # Convert to float32 for better precision
        image = image.astype(np.float32)
        
        if self.preserve_features:
            # Wavelet denoising for feature preservation
            denoised = denoise_wavelet(image, 
                                     wavelet='db1', 
                                     mode='soft',
                                     sigma=self.wavelet_sigma,
                                     multichannel=False,
                                     convert2ycbcr=False)
        else:
            # Traditional Gaussian denoising
            denoised = gaussian(image, sigma=self.gaussian_sigma)
            
        return denoised

    def enhance_contrast(self, image):
        """Enhance image contrast while preventing saturation"""
        p2, p98 = np.percentile(image, (2, 98))
        enhanced = np.clip((image - p2) / (p98 - p2), 0, 1)
        return enhanced

    def handle_convolution(self, galaxy_img, psf_kernel):
        """Improved PSF convolution with edge handling"""
        # Normalize and center PSF kernel
        psf_kernel = psf_kernel / np.sum(psf_kernel)
        
        # Pad image to prevent edge effects
        pad_width = psf_kernel.shape[0] // 2
        padded_img = np.pad(galaxy_img, pad_width, mode='reflect')
        
        # Apply convolution
        convolved = convolve2d(padded_img, psf_kernel, mode='valid', boundary='wrap')
        
        return convolved

    def extract_psf_kernel(self, star_img):
        """Enhanced PSF kernel extraction"""
        # Find the center of the star
        h, w = star_img.shape
        center_h, center_w = h // 2, w // 2
        
        # Dynamic kernel size based on star FWHM
        intensity_profile = star_img[center_h, :]
        fwhm = np.sum(intensity_profile > np.max(intensity_profile) * 0.5)
        kernel_size = max(int(fwhm * 2), 16)  # At least 16 pixels
        
        # Extract and normalize kernel
        k_start_h = center_h - kernel_size // 2
        k_start_w = center_w - kernel_size // 2
        kernel = star_img[k_start_h:k_start_h + kernel_size,
                        k_start_w:k_start_w + kernel_size]
        
        return self.normalize_image(kernel)

    def handle_pixelization(self, image, kernel_size):
        """Improved pixelization with feature preservation"""
        # Calculate target size based on kernel
        target_size = max(kernel_size // 2, 56)  # Minimum size of 56
        
        # Use area interpolation for downsampling
        resized = cv2.resize(image, (target_size, target_size), 
                           interpolation=cv2.INTER_AREA)
        
        if self.preserve_features:
            # Apply gentle smoothing to prevent aliasing
            resized = gaussian(resized, sigma=0.5)
        
        return resized

    def normalize_image(self, image):
        """Enhanced normalization with outlier handling"""
        # Remove outliers
        p1, p99 = np.percentile(image, (1, 99))
        clipped = np.clip(image, p1, p99)
        
        # Normalize
        normalized = (clipped - np.min(clipped)) / (np.max(clipped) - np.min(clipped))
        return normalized

    def process_single_image(self, galaxy_img, star_img):
        """Process a single galaxy-star image pair with enhanced quality"""
        # Convert to grayscale if needed
        if len(galaxy_img.shape) == 3:
            galaxy_img = np.mean(galaxy_img, axis=0)
            star_img = np.mean(star_img, axis=0)

        # Remove background
        galaxy_img = self.remove_background(galaxy_img)
        
        # Extract PSF kernel
        psf_kernel = self.extract_psf_kernel(star_img)
        
        # Denoise while preserving features
        galaxy_img = self.denoise_image(galaxy_img)
        
        # Apply atmospheric effects
        galaxy_img = self.handle_convolution(galaxy_img, psf_kernel)
        
        # Handle pixelization
        kernel_size = psf_kernel.shape[0]
        galaxy_img = self.handle_pixelization(galaxy_img, kernel_size)
        
        # Enhance contrast
        galaxy_img = self.enhance_contrast(galaxy_img)
        
        return self.normalize_image(galaxy_img)

    def process_batch(self, batch):
        """Process a batch of images with enhanced quality"""
        galaxies = batch['galaxy'].cpu().numpy()
        stars = batch['star'].cpu().numpy()
        
        processed_galaxies = []
        processed_stars = []
        
        for galaxy, star in zip(galaxies, stars):
            # Convert from RGB to grayscale if necessary
            if len(galaxy.shape) == 3:
                galaxy = np.mean(galaxy, axis=0)
                star = np.mean(star, axis=0)
            
            processed_galaxy = self.process_single_image(galaxy, star)
            processed_star = self.normalize_image(star)
            
            processed_galaxies.append(processed_galaxy)
            processed_stars.append(processed_star)
        
        return {
            'processed_galaxies': torch.tensor(processed_galaxies).to(self.device),
            'processed_stars': torch.tensor(processed_stars).to(self.device),
            'paths': {
                'galaxy': batch['galaxy_path'],
                'star': batch['star_path']
            }
        }