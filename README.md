# JEPA LLM Sandwich

A modular framework for injecting Joint-Embedding Predictive Architecture (JEPA) concepts into frozen Large Language Models. This project trains a lightweight "JEPA Predictor" to steer the LLM's continuous latent representations toward a reasoning trajectory, allowing the model to "think" in latent space rather than relying strictly on autoregressive token-based Chain-of-Thought (CoT).

## The Approach

Traditional reasoning in LLMs relies on generating explicit step-by-step text tokens (Chain-of-Thought). While effective, this is computationally expensive at inference time and bottlenecks the model's thinking speed to the rate of token generation. 

The **JEPA LLM Sandwich** takes a different approach:
1. **Model Splitting:** A frozen base LLM (e.g., Gemma, Qwen, SmolLM) is intercepted at a specific intermediate layer (`split_layer`).
2. **Latent Space Intervention:** During the prompt prefill phase, a tiny trainable network (the JEPA Predictor) processes the hidden states and injects a continuous "thought vector" back into the model.
3. **Supervised Latent Trajectory Optimization:** We extract the ground-truth target latent representations by passing a full reasoning trace (Prompt + CoT + Solution) through the frozen model. We then train our Predictor to shift the prompt-only latents directly toward this target reasoning manifold.
4. **Reinforcement Learning Refinement:** In the second stage, we fine-tune the predictor directly on correctness and token efficiency using Policy Gradients (REINFORCE) with a baseline reward function.
5. **Generation:** At inference time, the Predictor shifts the model's mental state in a single forward pass, guiding it to output the correct final answer directly or altering its generation behavior without needing to generate the intermediate reasoning tokens.

## Based On

This architecture is heavily inspired by:
- **Joint-Embedding Predictive Architecture (JEPA):** Proposed by Yann LeCun (Meta AI) for self-supervised learning, aiming to learn highly abstract representations by predicting continuous latents rather than reconstructing raw data (pixels or tokens).
- **Latent Space Reasoning:** Concepts seen in recent works like *Quiet-STaR* and various "Think Before You Speak" architectures, which argue that LLMs should perform planning and reasoning in their continuous hidden states for better efficiency and generalization.

## Project Structure

The project has been refactored into a clean, modular architecture:
- `model.py`: Contains the `JEPALangSandwich` wrapper, MLP/Transformer predictor blocks, and dynamic checkpoint loading.
- `train.py`: The supervised training engine, handling datasets (`simplescaling/s1K-1.1`), gradient accumulation, and MSE/Cosine loss minimization.
- `train_rl.py`: The reinforcement learning training engine, optimizing the predictor using REINFORCE based on correct answers and token length savings on `openai/gsm8k`.
- `eval.py`: Evaluation utilities for calculating continuous validation losses.
- `manual_test.py`: A CLI interface for qualitatively comparing text generation with the Predictor ON vs. OFF.
- `benchmark.py`: A quantitative benchmarking script measuring accuracy (match rate), generation speed, and tokens saved.
- `main.py`: A backward-compatible router that delegates CLI execution to supervised training (`train`), RL training (`train_rl`), or manual testing.

---

## How to Use

### 1. Installation

Ensure you have PyTorch and the required Hugging Face libraries installed:
```bash
pip install -r requirements.txt
```

### 2. Supervised Training (Stage 1)

Train the predictor to align its outputs with reasoning trajectories from the dataset. You can choose either an MLP or Transformer architecture, along with different loss functions.

```bash
# Supervised training using a Transformer predictor and combined MSE + Cosine loss:
python main.py --mode train --model_name HuggingFaceTB/SmolLM3-3B --predictor_type transformer --loss_type combined --epochs 3
```

**Training Arguments:**
- `--predictor_type`: Type of predictor architecture: `mlp` (default) or `transformer`.
- `--loss_type`: Loss calculation method: `mse` (default), `cosine` (1 - CosineSimilarity), or `combined` (MSE + Cosine).
- `--dataset_name`: Reasoning dataset (defaults to `gsm8k`).
- `--predictor_path`: File path to save/load weights (default: `jepa_predictor.pt`).

*Note: Checkpoints automatically save the predictor architecture type in their metadata. When loading weights later, the wrapper dynamically detects and instantiates the correct architecture type.*

### 3. Reinforcement Learning Training (Stage 2)

Optimize the predictor directly for question correctness and token efficiency using Policy Gradients (REINFORCE) on the `gsm8k` dataset.

```bash
# RL training of a Transformer predictor on GSM8K correctness:
python main.py --mode train_rl --model_name HuggingFaceTB/SmolLM3-3B --predictor_type transformer --epochs 3 --predictor_path jepa_predictor_rl.pt
```

**RL Arguments:**
- `--num_samples`: Max training samples per epoch.
- `--val_samples`: Validation dataset size for checkpoiting.
- `--max_new_tokens`: Generation budget limit for rewards.
- `--temperature`: Stochastic policy exploration temperature (default: `0.7`).

### 4. Quantitative Benchmarking

Measure token efficiency, generation speed, and correctness (match rate) between the base model and the JEPA-sandwich model:

```bash
python benchmark.py --model_name HuggingFaceTB/SmolLM3-3B --predictor_path jepa_predictor_rl.pt --num_samples 20
```

### 5. Manual Testing

To see the qualitative difference the JEPA predictor makes on the model's generation, run:

```bash
python main.py --mode test --model_name HuggingFaceTB/SmolLM3-3B --predictor_path jepa_predictor_rl.pt
```
