import argparse
import sys
import os
import torch
import random

from model import JEPALangSandwich

def main():
    parser = argparse.ArgumentParser(description="JEPA LLM Sandwich Manual Generation & Verification")
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM3-3B", help="Name or path of the base LLM")
    parser.add_argument("--split_layer", type=int, default=18, help="Layer index to insert the predictor")
    parser.add_argument("--predictor_path", type=str, default="jepa_predictor.pt", help="Path to load/save the predictor")
    parser.add_argument("--prompt", type=str, default="Solve the following puzzle step-by-step: If external features shift, look internal.", help="Prompt to generate text for")
    parser.add_argument("--max_new_tokens", type=int, default=65536, help="Maximum new tokens to generate")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    jepa_llm = JEPALangSandwich(
        model_name=args.model_name,
        split_layer=args.split_layer,
        predictor_path=args.predictor_path if os.path.exists(args.predictor_path) else None
    )

    print(f"\nPrompt: {args.prompt}")
    print("\nRunning generation with Predictor OFF (Base Model)...")
    out_without = jepa_llm.generate_text(args.prompt, max_new_tokens=args.max_new_tokens, use_predictor=False)
    print(f"Output:\n{out_without}")

    print("\nRunning generation with Predictor ON (JEPA Guided)...")
    out_with = jepa_llm.generate_text(args.prompt, max_new_tokens=args.max_new_tokens, use_predictor=True)
    print(f"Output:\n{out_with}")

if __name__ == "__main__":
    main()
