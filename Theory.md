**Mapping Dark Matter Using GANs and Deep Learning**

### **Abstract**

This project focuses on mapping the distribution of dark matter using weak gravitational lensing and Generative Adversarial Networks (GANs). Gravitational lensing distorts the shapes of galaxies due to the gravitational influence of dark matter. By measuring these distortions, we aim to reconstruct the underlying dark matter density distribution. The project employs deep learning, specifically GANs, to map the mass density from observed ellipticity data while accounting for noise and Point Spread Function (PSF) distortions. This report details the methodology, key challenges, and implementation strategies.

---

### **1. Introduction**

Dark matter is an invisible and elusive component of the universe that accounts for about 27% of its mass-energy content. Although it does not interact with light directly, its gravitational influence causes observable distortions in the shapes of background galaxies, a phenomenon known as weak gravitational lensing. Mapping the distribution of dark matter provides critical insights into the structure and evolution of the universe.

The primary goal of this project is to develop a model that reconstructs dark matter mass density maps from galaxy ellipticity data. The methodology combines image analysis, PSF corrections, and deep learning, specifically using GANs, to predict the underlying mass distribution.

---

### **2. Background**

#### **2.1 Weak Gravitational Lensing**

Weak lensing occurs when dark matter bends light from background galaxies, causing small distortions in their observed shapes. This effect is measured in terms of ellipticity, which represents the deviation of a galaxy’s shape from being circular. By analyzing ellipticity fields across many galaxies, scientists can infer the lensing signal and reconstruct the dark matter distribution.

#### **2.2 Point Spread Function (PSF)**

The PSF represents distortions introduced by telescope optics, atmospheric conditions, and detector noise. PSF corrections are essential for accurately measuring galaxy shapes and weak lensing signals. Star images, which serve as point sources, are used to model and remove PSF effects.

#### **2.3 Generative Adversarial Networks (GANs)**

GANs are a class of deep learning models consisting of a generator and a discriminator. The generator produces synthetic outputs (e.g., mass density maps) from input data (e.g., ellipticity fields), while the discriminator evaluates their realism. This adversarial training framework helps the generator produce high-quality outputs that resemble real data.

---

### **3. Methodology**

#### **3.1 Input Data**

- **Ellipticity Data:** Measured galaxy shapes (γ₁, γ₂) representing weak lensing distortions.
- **Redshift Data:** Used for tomographic reconstruction to map dark matter in 3D.
- **PSF-Star Pairs:** Data to model and correct PSF distortions.

#### **3.2 Data Preprocessing**

1. **Noise Reduction:**
   - Gaussian filtering and wavelet-based denoising to reduce high-frequency noise.
   - Use of denoising autoencoders to improve image quality.
2. **PSF Deconvolution:**
   - Applied Wiener filtering and Richardson-Lucy deconvolution to correct PSF effects.

#### **3.3 Model Design**

**Generator:**

- Input: Ellipticity field (e1, e2) or shear components (γ₁, γ₂).
- Architecture: Encoder-decoder with skip connections (e.g., UNet).
- Output: Predicted mass density map (κ).

**Discriminator:**

- Input: Real and generated mass density maps.
- Output: Probability that the input is real or generated.

#### **3.4 Loss Functions**

1. **Adversarial Loss:** Ensures the generator produces realistic mass density maps:

   \[
   \mathcal{L}\_{\text{GAN}} = -\mathbb{E}[\log(D(G(e)))]
   \]

2. **Reconstruction Loss:** Minimizes the difference between true and generated maps:

   \[
   \mathcal{L}_{\text{reconstruction}} = \| \kappa_{\text{true}} - G(e) \|\_2^2
   \]

3. **Regularization Loss:** Adds smoothness constraints to the mass density map:

   \[
   \mathcal{L}\_{\text{regularization}} = \lambda \|\nabla^2 G(e)\|
   \]

#### **3.5 Training Procedure**

1. Train the generator to produce mass density maps from ellipticity fields.
2. Train the discriminator to distinguish between real and generated maps.
3. Alternate updates for generator and discriminator to optimize the adversarial framework.

---

### **4. Implementation**

#### **4.1 Software and Tools**

- **Programming Language:** Python
- **Libraries:** PyTorch, NumPy, SciPy, Astropy
- **Visualization:** Matplotlib, Plotly

#### **4.2 GAN Training Steps**

1. Load simulated ellipticity and mass density data.
2. Preprocess images to normalize values and reduce noise.
3. Train the GAN using Adam optimizer with a learning rate of 0.0002.
4. Validate the model on unseen data to ensure generalization.

#### **4.3 Metrics for Evaluation**

- **Mean Squared Error (MSE):** Measures the difference between true and predicted maps.
- **Structural Similarity Index (SSIM):** Evaluates the perceptual similarity of images.
- **Precision-Recall:** Assesses the discriminator’s performance in distinguishing real and generated maps.

---

### **5. Results and Discussion**

#### **5.1 Model Performance**

- The generator successfully reconstructed high-quality mass density maps, capturing both local and global dark matter distributions.
- Regularization ensured smooth and physically plausible maps.

#### **5.2 Challenges**

- Noise in ellipticity data posed significant challenges for accurate reconstruction.
- PSF modeling and corrections were critical but computationally intensive.

#### **5.3 Comparison with Traditional Methods**

GAN-based reconstruction outperformed traditional Fourier-based inversion methods in handling noise and preserving fine-grained details.

---

### **6. Future Work**

1. Incorporate real observational data from surveys like LSST and Euclid.
2. Extend the model to 3D tomographic mapping using redshift data.
3. Use hybrid models combining GANs with Bayesian inference for uncertainty estimation.
4. Investigate transfer learning to adapt the model to different datasets.

---

### **7. Conclusion**

This project demonstrated the potential of GANs for mapping dark matter from weak gravitational lensing data. By leveraging deep learning, we reconstructed detailed and realistic mass density maps, advancing our understanding of the universe’s dark matter distribution. The methodology holds promise for future applications in large-scale cosmological surveys.

---

### **References**

1. Einstein, A. (1936). Lens-like action of a star by the deviation of light in the gravitational field.
2. GREAT08 and GREAT10 Handbooks.
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning.
4. LSST Science Collaboration (2009). LSST Science Book.
