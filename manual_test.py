import argparse
import sys
import os
import torch
import random

from model import JEPALangSandwich, project_thoughts_pca

def main():
    parser = argparse.ArgumentParser(description="JEPA LLM Sandwich Manual Generation & Verification")
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM3-3B", help="Name or path of the base LLM")
    parser.add_argument("--split_layer", type=str, default="18", help="Layer index or indices (comma-separated) to insert the predictor")
    parser.add_argument("--predictor_path", type=str, default="jepa_predictor.pt", help="Path to load/save the predictor")
    parser.add_argument("--prompt", type=str, default="Solve the following puzzle step-by-step: If external features shift, look internal.", help="Prompt to generate text for")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Maximum new tokens to generate")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling instead of greedy decoding")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top-p")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--num_tasks", type=int, default=None, help="Number of tasks for task-conditioned embeddings")
    parser.add_argument("--task_id", type=int, default=None, help="Specific task ID for conditioning")
    parser.add_argument("--target_cot", type=str, default=None, help="Optional target CoT sequence to measure similarity against")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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

    generation_kwargs = {
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }

    print(f"\nPrompt: {args.prompt}")
    print("\nRunning generation with Predictor OFF (Base Model)...")
    out_without = jepa_llm.generate_text(
        args.prompt, 
        max_new_tokens=args.max_new_tokens, 
        use_predictor=False, 
        task_id=args.task_id, 
        **generation_kwargs
    )
    print(f"Output:\n{out_without}")

    print("\nRunning generation with Predictor ON (JEPA Guided)...")
    out_with = jepa_llm.generate_text(
        args.prompt, 
        max_new_tokens=args.max_new_tokens, 
        use_predictor=True, 
        task_id=args.task_id, 
        **generation_kwargs
    )
    print(f"Output:\n{out_with}")

    # Interpretability utilities check
    print("\n" + "="*45)
    print("        INTERPRETABILITY & UTILITIES        ")
    print("="*45)
    
    # 1. Extract thought vectors
    print("Extracting injected thought vectors...")
    thoughts = jepa_llm.extract_thought_vectors(args.prompt, task_id=args.task_id)
    for layer, vec in thoughts.items():
        norm = vec.float().norm().item()
        print(f"  Layer {layer} Predictor Output Shape: {list(vec.shape)} | L2 Norm: {norm:.4f}")
        
    # 2. PCA project if we have multiple vectors (e.g. over sequence tokens)
    if len(thoughts) > 0:
        first_layer = list(thoughts.keys())[0]
        # shape of vec is [1, seq_len, hidden_size]
        vec_seq = thoughts[first_layer][0] # [seq_len, hidden_size]
        if vec_seq.shape[0] > 1:
            projected = project_thoughts_pca(vec_seq, n_components=2)
            print(f"  Projected layer {first_layer} sequence of length {vec_seq.shape[0]} using PCA (SVD):")
            print(f"    First 3 token coordinates: {projected[:3].tolist()}")

    # 3. Calculate target similarity if target_cot is provided
    if args.target_cot:
        print(f"\nAnalyzing thought similarity with target CoT trajectory: '{args.target_cot}'")
        try:
            similarities = jepa_llm.analyze_thought_similarity(args.prompt, args.target_cot, task_id=args.task_id)
            for layer, sim in similarities.items():
                print(f"  Layer {layer} Cosine Similarity to target trajectory: {sim:.4f}")
        except Exception as e:
            print(f"  Failed to compute target similarity: {e}")
            
        print("\nAnalyzing layer-by-layer trajectory similarity vs. target CoT:")
        try:
            traj_sims = jepa_llm.analyze_layer_trajectory_similarity(args.prompt, args.target_cot, task_id=args.task_id)
            for lyr in sorted(traj_sims.keys()):
                if lyr in jepa_llm.split_layers or lyr in [0, len(traj_sims)//2, len(traj_sims)-1]:
                    marker = " <-- SPLIT LAYER" if lyr in jepa_llm.split_layers else ""
                    print(f"  Layer {lyr:2d} Cosine Similarity: {traj_sims[lyr]:.4f}{marker}")
        except Exception as e:
            print(f"  Failed to compute layer trajectory similarity: {e}")

    # 4. Measure logit bias
    print("\nMeasuring output logit bias with predictor ON vs OFF:")
    try:
        bias_metrics = jepa_llm.measure_logit_bias(args.prompt, task_id=args.task_id)
        print(f"  KL Divergence of next-token distribution: {bias_metrics['kl_divergence']:.4f}")
        print(f"  Cosine Similarity of logits:              {bias_metrics['cosine_similarity_logits']:.4f}")
        print("  Top-3 next tokens (Without Predictor):")
        for tok, prob in bias_metrics['top_tokens_off'][:3]:
            print(f"    - {tok!r}: {prob*100:.2f}%")
        print("  Top-3 next tokens (With Predictor):")
        for tok, prob in bias_metrics['top_tokens_on'][:3]:
            print(f"    - {tok!r}: {prob*100:.2f}%")
    except Exception as e:
        print(f"  Failed to measure logit bias: {e}")
            
    print("="*45)

if __name__ == "__main__":
    main()
