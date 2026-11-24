# Facet: looped: speculative

## Drafter taps count logical visits

Check the drafter's GGUF metadata for its indexing convention. Inkling-2-Mini's DSpark indexes logical visits 5, 23, 41
and 48 of 48. A tap inside the looped range selects a specific pass of a shared block. Map each tap through the loop
schedule when wiring the capture kernels.
