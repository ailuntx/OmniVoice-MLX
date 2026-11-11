# MLX speedups

Opt-in settings that make MLX generation faster on Apple Silicon. Defaults are unchanged; turn them on with keyword arguments.

```python
audio = model.generate(
    text=sentence, language="Bengali", voice_clone_prompt=voice,
    num_step=16, prefix_refresh=8, uncond_reuse=4,
)
```

## Results

Bengali voice cloning, 8-bit model, Apple M2 with 16 GB, one sentence per `generate` call. RTF = generation time / audio length (lower is faster).

| Settings | RTF |
|---|---|
| `num_step=24, prefix_refresh=8` | 0.95 |
| `num_step=16, prefix_refresh=8` | 0.61 |
| `num_step=16, prefix_refresh=8, uncond_reuse=2` | 0.50 |
| `num_step=16, prefix_refresh=8, uncond_reuse=4` | 0.41 |

The `uncond_reuse` rows are medians of an interleaved A/B run in one process (3 reps each), because single runs on a busy machine varied by 2x. Listening tests on Bengali prose found no clear quality loss at `uncond_reuse=4`; check on your own language and voice.

## What each change does

- **Fused kernels and projections.** Q/K/V and gate/up projections run as single matmuls, cutting kernel launches per layer.
- **Prefix K/V cache (`prefix_refresh`).** The text and reference-audio tokens dominate the sequence but barely change between unmasking steps. Their K/V is cached on a full pass, and the steps in between run only the target-audio suffix. A full pass every N steps keeps the cache in sync with newly unmasked audio. Batch size 1 only.
- **Unconditional-row reuse (`uncond_reuse`).** Classifier-free guidance runs a conditional and an unconditional row every step. On suffix steps the unconditional logits are recomputed only every N steps and reused in between, so most steps run batch 1 instead of 2. Full passes always compute both rows.
- **Sampling on the GPU.** CFG log-softmax, argmax and confidence scores run in MLX; only small `(1, C, T)` arrays come back to NumPy for top-k position selection. Position Gumbel noise now comes from `mx.random`, so seed MLX (`mx.random.seed`) for reproducible output. With `class_temperature > 0` sampling falls back to the NumPy path.

## Tried and dropped

- **Caching only masked positions.** Every frame stays partially masked until the last step (the layer penalty decides higher codebooks last), so no frame could be skipped: 0% saving.

## Not done yet

- Adaptive step count (stop once all positions are confident).
- Batch size > 1 with `prefix_refresh`.
