import argparse
import os
import random
import time
import torch
import torch.nn as nn
from datasets import load_dataset
from model import JEPALangSandwich

def compute_reward(solution, generated_text, gen_tokens, max_new_tokens):
    """Computes reward based on solution match and token length savings."""
    sol = solution.strip().lower()
    gen = generated_text.strip().lower()
    
    if not sol:
        return 0.0
        
    match = sol in gen
    if match:
        # Base reward of 1.0 for correctness
        reward = 1.0
        # Efficiency incentive: extra reward for using fewer tokens
        # Maximum extra reward of +0.10
        if len(gen_tokens) > 0 and max_new_tokens > 0:
            efficiency = (max_new_tokens - len(gen_tokens)) / max_new_tokens
            reward += 0.10 * max(0.0, efficiency)
        return reward
    else:
        return 0.0

def evaluate_accuracy(jepa_llm, val_dataset, max_new_tokens, task_id=None):
    """Deterministically evaluates the model's accuracy on the validation set."""
    jepa_llm.predictor.eval()
    matches = 0
    total = len(val_dataset)
    
    for sample in val_dataset:
        question = sample.get("question", "")
        
        # Parse answer for gsm8k or solution for s1k
        if "answer" in sample:
            answer_text = sample.get("answer", "") or ""
            solution = answer_text.split("####", 1)[-1].strip() if "####" in answer_text else answer_text.strip()
        else:
            solution = sample.get("solution", "") or ""
            
        try:
            prompt = jepa_llm.tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            prompt = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
            
        with torch.no_grad():
            out_text = jepa_llm.generate_text(prompt, max_new_tokens=max_new_tokens, use_predictor=True, task_id=task_id)
            
        if solution.lower() in out_text.lower():
            matches += 1
            
    accuracy = (matches / max(1, total)) * 100
    return accuracy

def train_rl(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Parse split_layer string/list/int if provided
    split_layer_arg = args.split_layer
    if "," in split_layer_arg:
        split_layer_arg = [int(x.strip()) for x in split_layer_arg.split(",")]
    else:
        try:
            split_layer_arg = int(split_layer_arg)
        except ValueError:
            pass

    # Load sandwich wrapper
    jepa_llm = JEPALangSandwich(
        model_name=args.model_name,
        split_layer=split_layer_arg,
        predictor_path=args.predictor_path if os.path.exists(args.predictor_path) else None,
        predictor_type=args.predictor_type,
        num_tasks=args.num_tasks
    )

    optimizer = torch.optim.AdamW(jepa_llm.predictor.parameters(), lr=args.lr)

    # Load dataset
    print(f"Loading dataset: {args.dataset_name}...")
    if args.dataset_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main")
    else:
        dataset = load_dataset(args.dataset_name)

    if "train" not in dataset:
        raise ValueError("Dataset split 'train' not found.")

    full_dataset = list(dataset["train"])
    random.seed(args.seed)
    random.shuffle(full_dataset)

    # Split for train and validation
    val_size = min(len(full_dataset), args.val_samples)
    val_dataset = full_dataset[:val_size]
    train_dataset = full_dataset[val_size:]

    print(f"Dataset split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    # Initialize running baseline reward
    baseline = 0.0
    baseline_alpha = 0.95
    best_accuracy = -1.0

    for epoch in range(args.epochs):
        print(f"\n--- RL Epoch {epoch + 1}/{args.epochs} ---")
        jepa_llm.predictor.train()
        optimizer.zero_grad()
        
        epoch_rewards = []
        epoch_losses = []
        
        # Shuffle training set each epoch
        random.shuffle(train_dataset)
        
        # Limit epoch samples if requested
        epoch_samples = train_dataset[:args.num_samples] if args.num_samples else train_dataset
        
        for idx, sample in enumerate(epoch_samples):
            question = sample.get("question", "")
            
            if "answer" in sample:
                answer_text = sample.get("answer", "") or ""
                solution = answer_text.split("####", 1)[-1].strip() if "####" in answer_text else answer_text.strip()
            else:
                solution = sample.get("solution", "") or ""
                
            if not question or not solution:
                continue

            # Format prompt
            try:
                prompt = jepa_llm.tokenizer.apply_chat_template(
                    [{"role": "user", "content": question}],
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                prompt = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"

            prompt_ids = jepa_llm.tokenizer(prompt, return_tensors="pt").input_ids.to(jepa_llm.model.device)

            # 1. Rollout: Generate text under stochastic sampling policy (Predictor ON)
            try:
                gen_tokens, _ = jepa_llm.sample(
                    prompt_ids=prompt_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=50,
                    top_p=1.0,
                    use_predictor=True,
                    task_id=args.task_id
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
            
            if len(gen_tokens) == 0:
                continue
                
            generated_text = jepa_llm.tokenizer.decode(gen_tokens, skip_special_tokens=True)

            # 2. Compute reward
            reward = compute_reward(solution, generated_text, gen_tokens, args.max_new_tokens)
            epoch_rewards.append(reward)

            # 3. Policy Gradient Step (REINFORCE)
            try:
                sequence_log_prob = jepa_llm.compute_sequence_log_probs(
                    prompt_ids=prompt_ids,
                    gen_tokens=gen_tokens,
                    use_predictor=True,
                    task_id=args.task_id
                )
                
                # Policy Gradient Loss: - (Reward - Baseline) * LogProb
                advantage = reward - baseline
                loss = - advantage * sequence_log_prob
                
                scaled_loss = loss / args.grad_accum_steps
                scaled_loss.backward()
                
                epoch_losses.append(loss.item())
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue

            # Update running baseline
            baseline = baseline_alpha * baseline + (1.0 - baseline_alpha) * reward

            # Gradient Step
            if (idx + 1) % args.grad_accum_steps == 0 or (idx + 1) == len(epoch_samples):
                optimizer.step()
                optimizer.zero_grad()

            if (idx + 1) % 5 == 0 or (idx + 1) == len(epoch_samples):
                avg_r = sum(epoch_rewards[-5:]) / len(epoch_rewards[-5:])
                print(f"Sample {idx+1}/{len(epoch_samples)} | Reward: {reward:.2f} (Avg: {avg_r:.2f}) | Baseline: {baseline:.2f}", flush=True)

        # Epoch Metrics
        avg_reward = sum(epoch_rewards) / max(1, len(epoch_rewards))
        avg_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        print(f"Epoch {epoch+1} Completed | Average Reward: {avg_reward:.4f} | Average Loss: {avg_loss:.4f}")

        # Evaluation & Checkpointing
        print("Running validation accuracy check...")
        accuracy = evaluate_accuracy(jepa_llm, val_dataset, args.max_new_tokens, task_id=args.task_id)
        
        train_acc_subset = random.sample(train_dataset, min(len(train_dataset), args.val_samples))
        train_accuracy = evaluate_accuracy(jepa_llm, train_acc_subset, args.max_new_tokens, task_id=args.task_id)
        print(f"Training Accuracy (subset): {train_accuracy:.2f}% | Validation Accuracy: {accuracy:.2f}% (Best: {best_accuracy:.2f}%)")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            jepa_llm.save_predictor(args.predictor_path)
            print(f"New best validation accuracy! Saved checkpoint to {args.predictor_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JEPA LLM Sandwich RL training (REINFORCE)")
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM3-3B", help="Model name or path")
    parser.add_argument("--split_layer", type=str, default="18", help="Split layer index or indices (comma-separated)")
    parser.add_argument("--predictor_path", type=str, default="jepa_predictor_rl.pt", help="Path to save predictor")
    parser.add_argument("--predictor_type", type=str, default="mlp", choices=["mlp", "transformer", "trs"], help="Predictor type")
    parser.add_argument("--dataset_name", type=str, default="gsm8k", help="Dataset name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--grad_accum_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of training samples per epoch (None for all)")
    parser.add_argument("--val_samples", type=int, default=20, help="Number of validation samples")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Max output tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Stochastic policy temperature")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--num_tasks", type=int, default=None, help="Number of tasks for task-conditioned embeddings")
    parser.add_argument("--task_id", type=int, default=None, help="Specific task ID for conditioning during training/generation")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_rl(args)
