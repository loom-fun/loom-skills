# Diffusion on Strix Halo

Qualify default and bounded-workspace image presets separately on gfx1151. The latter can tile the
VAE at overlapping boundaries, so compare decoded pixels across the full frame. Time one resident
model at a time to avoid relocating weights into shared memory during a trial.
