import argparse
import sys
import os
import torch
import torch.nn as nn
from datasets import load_dataset
import random

from model import JEPALangSandwich
from eval import val_jepa_step, compute_loss

def train_jepa_step(sandwich_model, prompt_ids, target_ids, optimizer, loss_type="mse", grad_accum_steps=1, is_step=True):
    """Executes a single optimization step minimizing loss between predicted and target thoughts."""
    prompt_len = prompt_ids.shape[-1]
    full_input_ids = torch.cat([prompt_ids, target_ids], dim=-1)
    full_attention_mask = torch.ones_like(full_input_ids)
    
    sandwich_model.use_predictor = False
    with torch.no_grad():
        full_outputs = sandwich_model.model(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            output_hidden_states=True
        )
        target_latents = full_outputs.hidden_states[sandwich_model.split_layer + 1][:, prompt_len:, :]
        target_thought = target_latents.mean(dim=1)

    sandwich_model.use_predictor = True
    prompt_attention_mask = torch.ones_like(prompt_ids)
    prompt_outputs = sandwich_model.model(
        input_ids=prompt_ids,
        attention_mask=prompt_attention_mask,
        output_hidden_states=True
    )
    predicted_latents = prompt_outputs.hidden_states[sandwich_model.split_layer + 1]
    predicted_thought = predicted_latents[:, -1, :]

    loss = compute_loss(predicted_thought, target_thought, loss_type)
    scaled_loss = loss / grad_accum_steps
    scaled_loss.backward()

    if is_step:
        optimizer.step()
        optimizer.zero_grad()

    return loss.item()

def evaluate_accuracy(jepa_llm, samples, max_new_tokens=100):
    """Deterministically evaluates the model's accuracy (match rate) on a given list of samples."""
    jepa_llm.predictor.eval()
    matches = 0
    total = len(samples)
    
    for sample in samples:
        question = sample.get("question", "")
        
        # Parse solution / answer
        if "deepseek_thinking_trajectory" in sample:
            solution = sample.get("solution", "") or ""
        elif "answer" in sample:
            answer_text = sample.get("answer", "") or ""
            solution = answer_text.split("####", 1)[-1].strip() if "####" in answer_text else answer_text.strip()
        else:
            solution = sample.get("solution", "") or ""
            
        if not question or not solution:
            total -= 1
            continue
            
        try:
            prompt = jepa_llm.tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            prompt = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
            
        with torch.no_grad():
            out_text = jepa_llm.generate_text(prompt, max_new_tokens=max_new_tokens, use_predictor=True)
            
        if solution.lower() in out_text.lower():
            matches += 1
            
    accuracy = (matches / max(1, total)) * 100
    return accuracy

def train_jepa(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using main device: {device}")

    jepa_llm = JEPALangSandwich(
        model_name=args.model_name,
        split_layer=args.split_layer,
        predictor_path=args.predictor_path if os.path.exists(args.predictor_path) else None,
        predictor_type=args.predictor_type
    )

    optimizer = torch.optim.AdamW(jepa_llm.predictor.parameters(), lr=args.lr)

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
    
    val_size = int(len(full_dataset) * args.val_ratio)
    val_dataset = full_dataset[:val_size]
    train_dataset = full_dataset[val_size:]
    
    print(f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} val")
    
    best_val_loss = float("inf")
    
    sample_vis_prompts = [
        "Solve the following puzzle step-by-step: If external features shift, look internal.",
        "Calculate 25 * 45 + 12.",
        "What is the sum of the first 10 prime numbers?"
    ]

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---", flush=True)
        jepa_llm.predictor.train()
        
        epoch_loss = 0.0
        step_count = 0
        accumulated_loss = 0.0
        
        # Shuffle training set each epoch
        random.shuffle(train_dataset)
        
        for idx, sample in enumerate(train_dataset):
            question = sample.get("question", "")
            
            if "deepseek_thinking_trajectory" in sample:
                cot = sample.get("deepseek_thinking_trajectory", "") or ""
                solution = sample.get("solution", "") or ""
            elif "answer" in sample:
                answer_text = sample.get("answer", "") or ""
                if "####" in answer_text:
                    cot, solution = answer_text.split("####", 1)
                    cot = cot.strip()
                    solution = solution.strip()
                else:
                    cot = ""
                    solution = answer_text
            else:
                cot = ""
                solution = sample.get("solution", "") or ""
            
            if not question:
                continue
                
            try:
                prompt = jepa_llm.tokenizer.apply_chat_template(
                    [{"role": "user", "content": question}],
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                prompt = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
                
            target_answer = f"<think>\n{cot}\n</think>\n{solution}" if cot else solution
                
            prompt_ids = jepa_llm.tokenizer(prompt, return_tensors="pt").input_ids.to(jepa_llm.model.device)
            target_ids = jepa_llm.tokenizer(target_answer, return_tensors="pt", add_special_tokens=False).input_ids.to(jepa_llm.model.device)
            
            total_len = prompt_ids.shape[-1] + target_ids.shape[-1]
            if total_len > args.max_length:
                continue
                
            is_step = ((idx + 1) % args.grad_accum_steps == 0) or (idx + 1 == len(train_dataset))
            
            try:
                loss_val = train_jepa_step(
                    sandwich_model=jepa_llm,
                    prompt_ids=prompt_ids,
                    target_ids=target_ids,
                    optimizer=optimizer,
                    loss_type=args.loss_type,
                    grad_accum_steps=args.grad_accum_steps,
                    is_step=is_step
                )
                accumulated_loss += loss_val
                epoch_loss += loss_val
                step_count += 1
            except torch.cuda.OutOfMemoryError:
                print("CUDA OOM caught! Skipping sample...")
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue
                
            if is_step:
                avg_loss = accumulated_loss / args.grad_accum_steps
                if idx % (args.grad_accum_steps * 5) == 0 or idx < args.grad_accum_steps:
                    print(f"Step {idx}/{len(train_dataset)} | Running Loss: {avg_loss:.6f}", flush=True)
                accumulated_loss = 0.0

        avg_epoch_train_loss = epoch_loss / max(1, step_count)
        print(f"Epoch {epoch + 1} Training Loss: {avg_epoch_train_loss:.6f}", flush=True)
        
        jepa_llm.predictor.eval()
        val_loss_sum = 0.0
        val_count = 0
        
        print("Running validation...")
        for val_idx, val_sample in enumerate(val_dataset):
            val_question = val_sample.get("question", "")
            
            if "deepseek_thinking_trajectory" in val_sample:
                val_cot = val_sample.get("deepseek_thinking_trajectory", "") or ""
                val_solution = val_sample.get("solution", "") or ""
            elif "answer" in val_sample:
                answer_text = val_sample.get("answer", "") or ""
                if "####" in answer_text:
                    val_cot, val_solution = answer_text.split("####", 1)
                    val_cot = val_cot.strip()
                    val_solution = val_solution.strip()
                else:
                    val_cot = ""
                    val_solution = answer_text
            else:
                val_cot = ""
                val_solution = val_sample.get("solution", "") or ""
            
            if not val_question:
                continue
                
            try:
                val_prompt = jepa_llm.tokenizer.apply_chat_template(
                    [{"role": "user", "content": val_question}],
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                val_prompt = f"<start_of_turn>user\n{val_question}<end_of_turn>\n<start_of_turn>model\n"
                
            val_target = f"<think>\n{val_cot}\n</think>\n{val_solution}" if val_cot else val_solution
            val_prompt_ids = jepa_llm.tokenizer(val_prompt, return_tensors="pt").input_ids.to(jepa_llm.model.device)
            val_target_ids = jepa_llm.tokenizer(val_target, return_tensors="pt", add_special_tokens=False).input_ids.to(jepa_llm.model.device)
            
            total_len = val_prompt_ids.shape[-1] + val_target_ids.shape[-1]
            if total_len > args.max_length:
                continue
                
            try:
                loss_val = val_jepa_step(
                    sandwich_model=jepa_llm,
                    prompt_ids=val_prompt_ids,
                    target_ids=val_target_ids,
                    loss_type=args.loss_type
                )
                val_loss_sum += loss_val
                val_count += 1
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
                
        avg_val_loss = val_loss_sum / max(1, val_count)
        print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.6f}")
        
        # Calculate training and validation correctness accuracies
        print("Evaluating generation accuracies...")
        train_acc_subset = random.sample(train_dataset, min(len(train_dataset), 20))
        train_accuracy = evaluate_accuracy(jepa_llm, train_acc_subset, max_new_tokens=100)
        val_accuracy = evaluate_accuracy(jepa_llm, val_dataset[:20] if len(val_dataset) > 20 else val_dataset, max_new_tokens=100)
        print(f"Epoch {epoch + 1} | Training Accuracy (subset): {train_accuracy:.2f}% | Validation Accuracy: {val_accuracy:.2f}%")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            jepa_llm.save_predictor(args.predictor_path)
            print(f"New best validation loss: {best_val_loss:.6f}. Checkpoint saved.")
            
        print("\n--- Qualitative Output Comparison (Validation Sample) ---")
        vis_prompt = random.choice(sample_vis_prompts)
        print(f"Prompt: {vis_prompt}")
        with torch.no_grad():
            try:
                out_without = jepa_llm.generate_text(vis_prompt, max_new_tokens=80, use_predictor=False)
                out_with = jepa_llm.generate_text(vis_prompt, max_new_tokens=80, use_predictor=True)
                print(f"-> Predictor OFF (Base Model):\n{out_without}\n")
                print(f"-> Predictor ON  (JEPA Guided):\n{out_with}\n")
            except Exception as e:
                print(f"Could not generate text for visualization: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JEPA LLM Sandwich Training")
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM3-3B", help="Name or path of the base LLM")
    parser.add_argument("--split_layer", type=int, default=18, help="Layer index to insert the predictor")
    parser.add_argument("--predictor_path", type=str, default="jepa_predictor.pt", help="Path to load/save the predictor")
    parser.add_argument("--dataset_name", type=str, default="gsm8k", help="Dataset name")
    parser.add_argument("--predictor_type", type=str, default="mlp", choices=["mlp", "transformer", "trs"], help="Type of predictor architecture")
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "cosine", "combined"], help="Loss type for training")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--grad_accum_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation ratio")
    parser.add_argument("--max_length", type=int, default=2048, help="Max length")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_jepa(args)
