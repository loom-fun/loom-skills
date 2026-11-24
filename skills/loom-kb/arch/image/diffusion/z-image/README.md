# Z-Image

Z-Image-Turbo uses an eight-step single-stream denoiser with zero classifier-free guidance in
diffusers. Preserve the caption-row gate when testing batched attention. Convert the denoiser
latent to the VAE's domain using the checkpoint's scaling and shift before comparing decoder output.
