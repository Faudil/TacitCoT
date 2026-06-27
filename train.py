import argparse
import sys
import os
import torch
import torch.nn as nn
from datasets import load_dataset
import random

from model import JEPALangSandwich
from eval import compute_loss

def train_jepa_step(sandwich_model, prompt_ids, target_ids, optimizer, loss_type="mse", grad_accum_steps=1, is_step=True, task_id=None, cache=None, cache_key=None):
    """Executes a single optimization step minimizing loss between predicted and target thoughts."""
    device = next(sandwich_model.predictors.parameters()).device
    
    # 1. Retrieve or compute target thoughts (and prompt hidden state if single split layer)
    if cache is not None and cache_key is not None and cache_key in cache:
        cached_val = cache[cache_key]
        target_thoughts = {k: v.to(device) for k, v in cached_val["target_thoughts"].items()}
        prompt_hidden_state = cached_val["prompt_hidden_state"]
    else:
        prompt_len = prompt_ids.shape[-1]
        full_input_ids = torch.cat([prompt_ids, target_ids], dim=-1)
        full_attention_mask = torch.ones_like(full_input_ids)
        
        sandwich_model.use_predictor = False
        sandwich_model.current_task_id = task_id
        
        with torch.no_grad():
            full_outputs = sandwich_model.model(
                input_ids=full_input_ids,
                attention_mask=full_attention_mask,
                output_hidden_states=True
            )
            
            target_thoughts = sandwich_model.get_target_thoughts(full_outputs, full_input_ids, prompt_len)
            target_thoughts = {k: v.cpu() for k, v in target_thoughts.items()}
                
            prompt_hidden_state = None
            if len(sandwich_model.split_layers) == 1:
                prompt_attention_mask = torch.ones_like(prompt_ids)
                prompt_outputs = sandwich_model.model(
                    input_ids=prompt_ids,
                    attention_mask=prompt_attention_mask,
                    output_hidden_states=True
                )
                lyr = sandwich_model.split_layers[0]
                prompt_hidden_state = prompt_outputs.hidden_states[lyr + 1].cpu() # Keep on CPU
                
        if cache is not None and cache_key is not None:
            cache[cache_key] = {
                "target_thoughts": target_thoughts,
                "prompt_hidden_state": prompt_hidden_state
            }
        
        # Keep device-mapped dictionary for computation
        target_thoughts = {k: v.to(device) for k, v in target_thoughts.items()}

    # 2. Forward pass with trainable predictor parameters
    total_loss = 0.0
    
    if prompt_hidden_state is not None:
        # Single split layer: bypass model completely
        x = prompt_hidden_state.to(device)
        lyr = sandwich_model.split_layers[0]
        predictor = sandwich_model.predictors[str(lyr)]
        
        # Apply task conditioning if configured
        if sandwich_model.task_embeddings is not None:
            task_id_val = task_id if task_id is not None else 0
            if not isinstance(task_id_val, torch.Tensor):
                task_id_tensor = torch.tensor([task_id_val], device=device)
            else:
                task_id_tensor = task_id_val.to(device)
            
            batch_size = x.shape[0]
            if task_id_tensor.ndim == 1 and task_id_tensor.shape[0] == 1:
                task_id_tensor = task_id_tensor.expand(batch_size)
                
            task_emb_layer = sandwich_model.task_embeddings[str(lyr)]
            task_emb = task_emb_layer(task_id_tensor).unsqueeze(1)
            x = x + task_emb
            
        predictor_delta = predictor(x)
        gate = torch.sigmoid(sandwich_model.intervention_gates[str(lyr)])
        gated_delta = gate * predictor_delta
        predicted_thought = (prompt_hidden_state.to(device) + gated_delta)[:, -1, :]
        
        loss = compute_loss(predicted_thought, target_thoughts[str(lyr)], loss_type)
        total_loss = total_loss + loss
    else:
        # Multiple split layers: run model for prompt, but use cached targets
        sandwich_model.use_predictor = True
        sandwich_model.current_task_id = task_id
        
        prompt_attention_mask = torch.ones_like(prompt_ids)
        prompt_outputs = sandwich_model.model(
            input_ids=prompt_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True
        )
        
        for lyr in sandwich_model.split_layers:
            predicted_latents = prompt_outputs.hidden_states[lyr + 1]
            predicted_thought = predicted_latents[:, -1, :]
            
            loss = compute_loss(predicted_thought, target_thoughts[str(lyr)], loss_type)
            total_loss = total_loss + loss
            
        # Clean up task ID
        sandwich_model.current_task_id = None

    scaled_loss = total_loss / grad_accum_steps
    scaled_loss.backward()

    if is_step:
        # Gradient clipping to prevent runaway predictor updates
        trainable_params = [p for p in sandwich_model.predictors.parameters() if p.requires_grad]
        trainable_params += [p for p in sandwich_model.intervention_gates.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    return total_loss.item()


def val_jepa_step(sandwich_model, prompt_ids, target_ids, loss_type="mse", task_id=None, cache=None, cache_key=None):
    """Computes validation loss between predicted thoughts and targets using cached states."""
    device = next(sandwich_model.predictors.parameters()).device
    
    # 1. Retrieve or compute target thoughts (and prompt hidden state if single split layer)
    if cache is not None and cache_key is not None and cache_key in cache:
        cached_val = cache[cache_key]
        target_thoughts = {k: v.to(device) for k, v in cached_val["target_thoughts"].items()}
        prompt_hidden_state = cached_val["prompt_hidden_state"]
    else:
        prompt_len = prompt_ids.shape[-1]
        full_input_ids = torch.cat([prompt_ids, target_ids], dim=-1)
        full_attention_mask = torch.ones_like(full_input_ids)
        
        sandwich_model.use_predictor = False
        sandwich_model.current_task_id = task_id
        
        with torch.no_grad():
            full_outputs = sandwich_model.model(
                input_ids=full_input_ids,
                attention_mask=full_attention_mask,
                output_hidden_states=True
            )
            
            target_thoughts = sandwich_model.get_target_thoughts(full_outputs, full_input_ids, prompt_len)
            target_thoughts = {k: v.cpu() for k, v in target_thoughts.items()}
                
            prompt_hidden_state = None
            if len(sandwich_model.split_layers) == 1:
                prompt_attention_mask = torch.ones_like(prompt_ids)
                prompt_outputs = sandwich_model.model(
                    input_ids=prompt_ids,
                    attention_mask=prompt_attention_mask,
                    output_hidden_states=True
                )
                lyr = sandwich_model.split_layers[0]
                prompt_hidden_state = prompt_outputs.hidden_states[lyr + 1].cpu() # Keep on CPU
                
        if cache is not None and cache_key is not None:
            cache[cache_key] = {
                "target_thoughts": target_thoughts,
                "prompt_hidden_state": prompt_hidden_state
            }
        
        # Keep device-mapped dictionary for computation
        target_thoughts = {k: v.to(device) for k, v in target_thoughts.items()}

    # 2. Forward pass with predictor
    total_loss = 0.0
    
    if prompt_hidden_state is not None:
        # Single split layer: bypass model completely
        x = prompt_hidden_state.to(device)
        lyr = sandwich_model.split_layers[0]
        predictor = sandwich_model.predictors[str(lyr)]
        
        # Apply task conditioning if configured
        if sandwich_model.task_embeddings is not None:
            task_id_val = task_id if task_id is not None else 0
            if not isinstance(task_id_val, torch.Tensor):
                task_id_tensor = torch.tensor([task_id_val], device=device)
            else:
                task_id_tensor = task_id_val.to(device)
            
            batch_size = x.shape[0]
            if task_id_tensor.ndim == 1 and task_id_tensor.shape[0] == 1:
                task_id_tensor = task_id_tensor.expand(batch_size)
                
            task_emb_layer = sandwich_model.task_embeddings[str(lyr)]
            task_emb = task_emb_layer(task_id_tensor).unsqueeze(1)
            x = x + task_emb
            
        with torch.no_grad():
            predictor_delta = predictor(x)
            gate = torch.sigmoid(sandwich_model.intervention_gates[str(lyr)])
            gated_delta = gate * predictor_delta
            predicted_thought = (prompt_hidden_state.to(device) + gated_delta)[:, -1, :]
            
            loss = compute_loss(predicted_thought, target_thoughts[str(lyr)], loss_type)
            total_loss = total_loss + loss
    else:
        # Multiple split layers: run model for prompt, but use cached targets
        sandwich_model.use_predictor = True
        sandwich_model.current_task_id = task_id
        
        prompt_attention_mask = torch.ones_like(prompt_ids)
        with torch.no_grad():
            prompt_outputs = sandwich_model.model(
                input_ids=prompt_ids,
                attention_mask=prompt_attention_mask,
                output_hidden_states=True
            )
            
            for lyr in sandwich_model.split_layers:
                predicted_latents = prompt_outputs.hidden_states[lyr + 1]
                predicted_thought = predicted_latents[:, -1, :]
                
                loss = compute_loss(predicted_thought, target_thoughts[str(lyr)], loss_type)
                total_loss = total_loss + loss
            
        # Clean up task ID
        sandwich_model.current_task_id = None

    return total_loss.item()

def evaluate_accuracy(jepa_llm, samples, max_new_tokens=100, task_id=None):
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
            out_text = jepa_llm.generate_text(prompt, max_new_tokens=max_new_tokens, use_predictor=True, task_id=task_id)
            
        if solution.lower() in out_text.lower():
            matches += 1
            
    accuracy = (matches / max(1, total)) * 100
    return accuracy

def train_jepa(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using main device: {device}")

    # Parse split_layer string/list/int if provided
    split_layer_arg = args.split_layer
    if "," in split_layer_arg:
        split_layer_arg = [int(x.strip()) for x in split_layer_arg.split(",")]
    else:
        try:
            split_layer_arg = int(split_layer_arg)
        except ValueError:
            pass

    from target_extractors import ResponseTargetExtractor, ThinkingTargetExtractor
    if args.sandwich_type == "reasoning":
        target_extractor = ThinkingTargetExtractor()
    else:
        target_extractor = ResponseTargetExtractor()

    jepa_llm = JEPALangSandwich(
        model_name=args.model_name,
        split_layer=split_layer_arg,
        predictor_path=args.predictor_path if os.path.exists(args.predictor_path) else None,
        predictor_type=args.predictor_type,
        num_tasks=args.num_tasks,
        target_extractor=target_extractor,
        last_token_only=args.last_token_only
    )

    # Include both predictor and intervention gate parameters in optimizer
    trainable_params = list(jepa_llm.predictor.parameters()) + list(jepa_llm.intervention_gates.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    
    # Initialize cache for precomputed frozen weights results (lazy-caching on CPU)
    state_cache = {}
    if args.cache_path and os.path.exists(args.cache_path):
        print(f"Loading state cache from {args.cache_path}...", flush=True)
        try:
            loaded_cache = torch.load(args.cache_path, map_location="cpu")
            if isinstance(loaded_cache, dict) and "metadata" in loaded_cache and "data" in loaded_cache:
                meta = loaded_cache["metadata"]
                if (meta.get("model_name") == args.model_name and 
                        meta.get("split_layer") == args.split_layer and 
                        meta.get("sandwich_type") == args.sandwich_type):
                    state_cache = loaded_cache["data"]
                    print(f"Loaded {len(state_cache)} cached states successfully.", flush=True)
                else:
                    print("Warning: Cached states are incompatible with current model config (mismatch in model_name, split_layer, or sandwich_type). Starting with empty cache.", flush=True)
            else:
                state_cache = loaded_cache
                print(f"Loaded {len(state_cache)} cached states successfully (legacy format).", flush=True)
        except Exception as e:
            print(f"Failed to load cache: {e}. Starting with empty cache.", flush=True)

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
        cache_hits = 0
        cache_misses = 0
        
        # Shuffle training set each epoch
        random.shuffle(train_dataset)
        
        # Limit training samples if requested
        epoch_train_dataset = train_dataset[:args.num_samples] if args.num_samples else train_dataset
        
        for idx, sample in enumerate(epoch_train_dataset):
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
                
            if prompt in state_cache:
                cache_hits += 1
            else:
                cache_misses += 1
                
            target_answer = f"<think>\n{cot}\n</think>\n{solution}" if cot else solution
                
            prompt_ids = jepa_llm.tokenizer(prompt, return_tensors="pt").input_ids.to(jepa_llm.model.device)
            target_ids = jepa_llm.tokenizer(target_answer, return_tensors="pt", add_special_tokens=False).input_ids.to(jepa_llm.model.device)
            
            total_len = prompt_ids.shape[-1] + target_ids.shape[-1]
            if total_len > args.max_length:
                continue
                
            is_step = ((idx + 1) % args.grad_accum_steps == 0) or (idx + 1 == len(epoch_train_dataset))
            
            try:
                loss_val = train_jepa_step(
                    sandwich_model=jepa_llm,
                    prompt_ids=prompt_ids,
                    target_ids=target_ids,
                    optimizer=optimizer,
                    loss_type=args.loss_type,
                    grad_accum_steps=args.grad_accum_steps,
                    is_step=is_step,
                    task_id=args.task_id,
                    cache=state_cache,
                    cache_key=prompt
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
                if (idx + 1) % (args.grad_accum_steps * 5) == 0 or (idx + 1) <= args.grad_accum_steps or (idx + 1 == len(epoch_train_dataset)):
                    print(f"Step {idx}/{len(train_dataset)} | Running Loss: {avg_loss:.6f}", flush=True)
                accumulated_loss = 0.0

        avg_epoch_train_loss = epoch_loss / max(1, step_count)
        print(f"Epoch {epoch + 1} Training Loss: {avg_epoch_train_loss:.6f} | Cache Hits: {cache_hits} | Cache Misses: {cache_misses}", flush=True)
        
        jepa_llm.predictor.eval()
        val_loss_sum = 0.0
        val_count = 0
        val_cache_hits = 0
        val_cache_misses = 0
        
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
                
            if val_prompt in state_cache:
                val_cache_hits += 1
            else:
                val_cache_misses += 1
                
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
                    loss_type=args.loss_type,
                    task_id=args.task_id,
                    cache=state_cache,
                    cache_key=val_prompt
                )
                val_loss_sum += loss_val
                val_count += 1
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
                
        avg_val_loss = val_loss_sum / max(1, val_count)
        print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.6f} | Cache Hits: {val_cache_hits} | Cache Misses: {val_cache_misses}")
        
        # Calculate training and validation correctness accuracies
        print("Evaluating generation accuracies...")
        train_acc_subset = random.sample(train_dataset, min(len(train_dataset), 20))
        train_accuracy = evaluate_accuracy(jepa_llm, train_acc_subset, max_new_tokens=100, task_id=args.task_id)
        val_accuracy = evaluate_accuracy(jepa_llm, val_dataset[:20] if len(val_dataset) > 20 else val_dataset, max_new_tokens=100, task_id=args.task_id)
        print(f"Epoch {epoch + 1} | Training Accuracy (subset): {train_accuracy:.2f}% | Validation Accuracy: {val_accuracy:.2f}%")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            jepa_llm.save_predictor(args.predictor_path)
            print(f"New best validation loss: {best_val_loss:.6f}. Checkpoint saved.")
            
        if args.cache_path:
            print(f"Saving state cache to {args.cache_path}...", flush=True)
            try:
                cache_to_save = {
                    "metadata": {
                        "model_name": args.model_name,
                        "split_layer": args.split_layer,
                        "sandwich_type": args.sandwich_type
                    },
                    "data": state_cache
                }
                torch.save(cache_to_save, args.cache_path)
                print("Cache saved successfully.", flush=True)
            except Exception as e:
                print(f"Failed to save cache: {e}", flush=True)
            
        print("\n--- Qualitative Output Comparison (Validation Sample) ---")
        vis_prompt = random.choice(sample_vis_prompts)
        print(f"Prompt: {vis_prompt}")
        with torch.no_grad():
            try:
                out_without = jepa_llm.generate_text(vis_prompt, max_new_tokens=80, use_predictor=False, task_id=args.task_id)
                out_with = jepa_llm.generate_text(vis_prompt, max_new_tokens=80, use_predictor=True, task_id=args.task_id)
                print(f"-> Predictor OFF (Base Model):\n{out_without}\n")
                print(f"-> Predictor ON  (JEPA Guided):\n{out_with}\n")
            except Exception as e:
                print(f"Could not generate text for visualization: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JEPA LLM Sandwich Training")
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM3-3B", help="Name or path of the base LLM")
    parser.add_argument("--split_layer", type=str, default="18", help="Layer index or indices (comma-separated) to insert the predictor")
    parser.add_argument("--predictor_path", type=str, default="jepa_predictor.pt", help="Path to load/save the predictor")
    parser.add_argument("--dataset_name", type=str, default="gsm8k", help="Dataset name")
    parser.add_argument("--predictor_type", type=str, default="mlp", choices=["mlp", "transformer", "trs"], help="Type of predictor architecture")
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "cosine", "combined"], help="Loss type for training")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--grad_accum_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation ratio")
    parser.add_argument("--max_length", type=int, default=8192, help="Max length")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--num_tasks", type=int, default=None, help="Number of tasks for task-conditioned embeddings")
    parser.add_argument("--task_id", type=int, default=None, help="Specific task ID for conditioning during training/generation")
    parser.add_argument("--num_samples", type=int, default=1000, help="Limit number of training samples per epoch")
    parser.add_argument("--sandwich_type", type=str, default="standard", choices=["standard", "reasoning"], help="Type of target extractor strategy to inject")
    parser.add_argument("--cache_path", type=str, default="jepa_state_cache.pt", help="Path to load/save the precomputed state cache")
    parser.add_argument("--last_token_only", action=argparse.BooleanOptionalAction, default=True, help="Only modify the last token position during intervention (preserves KV cache)")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_jepa(args)
