# Safety LoRA

Train a tiny LoRA for efficient custom safety profiles, in this case Youth safety. Uses Qwen 3 as a base. Process: Generate/prepare a synthetic dataset, train a LoRA checkpoint with SFT, optionally run a second-stage GRPO-style refinement, then run quick evals to see how behavior changed.

Runs on cloud multi-GPUs using HF Accelerate, TRL PeFT.

What’s included:
- A data generation and cleanup pipeline that creates prompts/responses, scores them against a written safety policy, and rewrites unsafe responses into safer ones to form a training dataset.
- Dataset preparation utilities for splitting and rebalancing the data before training.
- Training code for SFT with LoRA, plus an optional GRPO stage that refines behavior using a reward signal.
- Lightweight evaluation and comparison tools to check checkpoints and measure basic behavior changes over time.
