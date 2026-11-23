# FLUX.2 Klein

Validate both the joint image/text transformer blocks and the subsequent single-stream blocks.
The distilled 4B checkpoint uses four denoising steps and a reference guidance of 1.0.
The FLUX.2 pipeline's returned latent is VAE-ready after unpatchifying and de-normalization.
