# Krea 2

The Turbo pipeline combines a Qwen3 text encoder and an eight-step denoiser. Keep reference
evidence per variant because the BF16 and INT8 denoisers use different arithmetic.
