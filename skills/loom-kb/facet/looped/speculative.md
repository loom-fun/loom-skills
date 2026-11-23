# Facet: looped: speculative

## Drafter taps count logical visits

A drafter fed by hidden states of the target names its taps by logical visit, not by physical block: Inkling-2-Mini's
DSpark taps visits 5, 23, 41 and 48 of 48, so a tap inside the looped range is a specific pass of a shared block. Map tap
indices through the loop schedule when wiring the capture kernels, and check the drafter's GGUF metadata for which count
it uses.
