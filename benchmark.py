import argparse
import os
import time
import torch
import random
from datasets import load_dataset
from model import JEPALangSandwich

def clean_match(solution, generated):
    """Checks if the ground-truth solution is contained within the generated text."""
    sol = solution.strip().lower()
    gen = generated.strip().lower()
    if not sol:
        return True
    return sol in gen

def run_benchmark(args):
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

    jepa_llm = JEPALangSandwich(
        model_name=args.model_name,
        split_layer=split_layer_arg,
        predictor_path=args.predictor_path if os.path.exists(args.predictor_path) else None,
        num_tasks=args.num_tasks
    )

    print(f"Loading dataset: {args.dataset_name}...")
    if args.dataset_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main")
    else:
        dataset = load_dataset(args.dataset_name)
        
    if "train" not in dataset:
        raise ValueError("Dataset split 'train' not found.")
    
    samples = list(dataset["train"])
    random.seed(args.seed)
    random.shuffle(samples)
    
    bench_samples = []
    for s in samples:
        question = s.get("question", "")
        if "solution" in s:
            solution = s.get("solution", "")
        elif "answer" in s:
            answer_text = s.get("answer", "")
            if "####" in answer_text:
                _, solution = answer_text.split("####", 1)
                solution = solution.strip()
            else:
                solution = answer_text
        else:
            solution = ""
            
        if question and solution:
            s["solution"] = solution
            bench_samples.append(s)
            
        if len(bench_samples) >= args.num_samples:
            break

    print(f"Benchmarking on {len(bench_samples)} samples...")

    metrics = {
        "off": {"tokens": [], "time": [], "matches": 0},
        "on": {"tokens": [], "time": [], "matches": 0}
    }
    
    kl_divergences = []
    cos_sims_logits = []

    generation_kwargs = {
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }

    for idx, sample in enumerate(bench_samples):
        question = sample["question"]
        solution = sample["solution"]
        
        try:
            prompt = jepa_llm.tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            prompt = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"

        print(f"\n[{idx+1}/{len(bench_samples)}] Prompt: {question[:80]}...")

        # Benchmark Predictor OFF
        t0 = time.time()
        out_off = jepa_llm.generate_text(
            prompt, 
            max_new_tokens=args.max_new_tokens, 
            use_predictor=False, 
            task_id=args.task_id, 
            **generation_kwargs
        )
        dt_off = time.time() - t0
        tokens_off = len(jepa_llm.tokenizer.encode(out_off, add_special_tokens=False))
        match_off = clean_match(solution, out_off)
        
        metrics["off"]["tokens"].append(tokens_off)
        metrics["off"]["time"].append(dt_off)
        if match_off:
            metrics["off"]["matches"] += 1

        # Benchmark Predictor ON
        t0 = time.time()
        out_on = jepa_llm.generate_text(
            prompt, 
            max_new_tokens=args.max_new_tokens, 
            use_predictor=True, 
            task_id=args.task_id, 
            **generation_kwargs
        )
        dt_on = time.time() - t0
        tokens_on = len(jepa_llm.tokenizer.encode(out_on, add_special_tokens=False))
        match_on = clean_match(solution, out_on)
        
        metrics["on"]["tokens"].append(tokens_on)
        metrics["on"]["time"].append(dt_on)
        if match_on:
            metrics["on"]["matches"] += 1

        # Measure logit bias
        try:
            bias = jepa_llm.measure_logit_bias(prompt, task_id=args.task_id)
            kl_divergences.append(bias["kl_divergence"])
            cos_sims_logits.append(bias["cosine_similarity_logits"])
        except Exception:
            pass

        print(f"  Predictor OFF: {tokens_off} tokens in {dt_off:.2f}s | Match: {match_off}")
        print(f"  Predictor ON : {tokens_on} tokens in {dt_on:.2f}s | Match: {match_on}")

    avg_tok_off = sum(metrics["off"]["tokens"]) / len(bench_samples)
    avg_tok_on = sum(metrics["on"]["tokens"]) / len(bench_samples)
    spared_tokens = avg_tok_off - avg_tok_on

    avg_time_off = sum(metrics["off"]["time"]) / len(bench_samples)
    avg_time_on = sum(metrics["on"]["time"]) / len(bench_samples)

    acc_off = (metrics["off"]["matches"] / len(bench_samples)) * 100
    acc_on = (metrics["on"]["matches"] / len(bench_samples)) * 100
    
    avg_kl = sum(kl_divergences) / len(kl_divergences) if kl_divergences else 0.0
    avg_cos_sim_logits = sum(cos_sims_logits) / len(cos_sims_logits) if cos_sims_logits else 1.0

    print("\n" + "="*45)
    print("              BENCHMARK RESULTS              ")
    print("="*45)
    print(f"Base Model (Predictor OFF):")
    print(f"  Average generated tokens: {avg_tok_off:.2f}")
    print(f"  Average generation time:  {avg_time_off:.2f}s")
    print(f"  Accuracy (Match Rate):    {acc_off:.1f}%")
    print(f"\nJEPA Sandwich (Predictor ON):")
    print(f"  Average generated tokens: {avg_tok_on:.2f}")
    print(f"  Average generation time:  {avg_time_on:.2f}s")
    print(f"  Accuracy (Match Rate):    {acc_on:.1f}%")
    print("-" * 45)
    print(f"Tokens Spared: {spared_tokens:.2f} tokens ({ (spared_tokens / max(1, avg_tok_off)) * 100:.1f}% savings)")
    print(f"\nInterpretability Analysis (Over Benchmark Prompts):")
    print(f"  Average logit KL-divergence (with vs. without predictor): {avg_kl:.4f}")
    print(f"  Average logit cosine similarity (with vs. without predictor): {avg_cos_sim_logits:.4f}")
    print("="*45)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JEPA LLM Sandwich Benchmark")
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM3-3B", help="Model name or path")
    parser.add_argument("--split_layer", type=str, default="18", help="Split layer index or indices (comma-separated)")
    parser.add_argument("--predictor_path", type=str, default="jepa_predictor.pt", help="Path to predictor weights")
    parser.add_argument("--dataset_name", type=str, default="gsm8k", help="Dataset name")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate on")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Max new tokens to generate")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling during generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top-p")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--num_tasks", type=int, default=None, help="Number of tasks for task-conditioned embeddings")
    parser.add_argument("--task_id", type=int, default=None, help="Specific task ID for conditioning")
    args = parser.parse_args()
    
    run_benchmark(args)
