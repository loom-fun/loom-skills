# Diffusion image pipelines

Map the text encoder, denoiser, scheduler and VAE independently against the pinned pipeline.
When validating pixel parity, hold the initial random tensor, latent layout, arithmetic dtype
and denoising schedule fixed. Check intermediate stage tensors to localize a mismatch.
