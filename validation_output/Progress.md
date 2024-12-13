# Dark Matter Mapping Pipeline Development Report

## Project Overview
Development of an image processing pipeline for galaxy and star images to support dark matter mapping, focusing on preprocessing astronomical data for machine learning applications.

## Initial Requirements
1. Handle atmospheric and telescope effects through convolution
2. Process pixelization (pixel size ≈ kernel size/2)
3. Remove both Poisson and Gaussian noise
4. Account for ~5% uncertainty in total light
5. Maintain relationship between galaxy images and PSF kernels
6. Process 40,000 training and 60,000 test images

## Development Process

### First Iteration
**Initial Approach**
- Basic data loader with minimal preprocessing
- Simple noise removal using Gaussian filtering
- Basic convolution without edge handling
- Direct pixelization without feature preservation

**Challenges Faced**
- Box-like artifacts in processed images
- Loss of fine galaxy structure
- Poor handling of background noise
- Inconsistent image normalization

### Second Iteration
**Improvements Attempted**
- Added ViT feature extractor
- Implemented basic wavelet denoising
- Added simple background subtraction
- Basic contrast enhancement

**Issues Encountered**
- API compatibility issues with scikit-image (`multichannel` parameter)
- High computational overhead from ViT
- Loss of important galaxy features
- Inadequate noise reduction

### Final Working Solution

**1. Enhanced Data Loading**
- Implemented robust error handling
- Added proper file path validation
- Improved image pair matching
- Added configurable normalization

**2. Improved Image Preprocessing**
Successfully implemented:
- Multi-scale background subtraction
- Two-pass denoising strategy
- Adaptive contrast enhancement
- Dynamic PSF kernel extraction

**3. Feature Preservation Techniques**
Effective methods:
- Wavelet denoising with `db1` wavelet
- Unsharp masking for feature enhancement
- Local adaptive contrast
- Edge-preserving smoothing

**4. Pipeline Validation**
Created comprehensive validation system:
- Batch processing verification
- Shape and value range checks
- Visual comparison tools
- Detailed logging

## Key Findings

### What Worked Well
1. **Noise Reduction**
   - Combined wavelet and Gaussian denoising
   - Adaptive noise thresholding
   - Multi-scale background removal

2. **Feature Preservation**
   - Two-pass denoising approach
   - Dynamic kernel size selection
   - Local contrast enhancement

3. **PSF Handling**
   - Improved center-of-mass calculation
   - Better kernel extraction
   - Enhanced edge handling in convolution

### What Didn't Work
1. **Attempted Approaches**
   - Single-pass Gaussian denoising (too aggressive)
   - Fixed kernel sizes (inflexible)
   - Global contrast enhancement (lost local details)
   - ViT feature extraction (computationally expensive)

2. **Technical Issues**
   - API compatibility problems
   - Memory issues with large batches
   - Loss of fine structure in early versions

## Current State
The pipeline successfully:
1. Loads and validates image pairs
2. Removes background noise while preserving features
3. Handles PSF effects appropriately
4. Maintains proper image scaling and normalization
5. Provides clear visualization of results

## Future Improvements
Potential areas for enhancement:
1. Further optimization of denoising parameters
2. More sophisticated PSF modeling
3. GPU acceleration for faster processing
4. Enhanced feature preservation techniques

## Conclusion
The current implementation provides a solid foundation for dark matter mapping, with effective noise reduction and feature preservation. While there's room for optimization, the pipeline is now ready for training applications.