import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_decoder_layers(model):
    """Dynamically locate the list of decoder layers in the model structure."""
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        return model.model.language_model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError("Could not find decoder layers in the model structure.")

class JEPALangSandwich(nn.Module):
    def __init__(self, model_name: str, split_layer=18, predictor_path=None, predictor_type="mlp", num_tasks=None):
        super().__init__()
        print(f"Loading frozen base model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto"
        )

        for param in self.model.parameters():
            param.requires_grad = False

        self.layers = get_decoder_layers(self.model)
        num_layers = len(self.layers)

        # Normalize split layers
        resolved_split_layers = None
        resolved_predictor_type = predictor_type

        # Check metadata from checkpoint first if available to resolve predictor type and split layers
        if predictor_path and os.path.exists(predictor_path):
            try:
                # Need map_location="cpu" to inspect
                checkpoint = torch.load(predictor_path, map_location="cpu")
                if isinstance(checkpoint, dict):
                    if "predictor_type" in checkpoint:
                        resolved_predictor_type = checkpoint["predictor_type"]
                        print(f"Detected predictor type '{resolved_predictor_type}' from checkpoint metadata.")
                    if "split_layers" in checkpoint:
                        resolved_split_layers = checkpoint["split_layers"]
                        print(f"Detected split layers '{resolved_split_layers}' from checkpoint metadata.")
            except Exception as e:
                print(f"Note: Could not inspect checkpoint metadata: {e}")

        # Configure split layers
        if resolved_split_layers is not None:
            self.split_layers = resolved_split_layers
        else:
            if isinstance(split_layer, str):
                if "," in split_layer:
                    self.split_layers = [int(x.strip()) for x in split_layer.split(",")]
                else:
                    self.split_layers = [int(split_layer)]
            elif isinstance(split_layer, int):
                self.split_layers = [split_layer]
            elif isinstance(split_layer, (list, tuple)):
                self.split_layers = list(split_layer)
            elif split_layer is None:
                self.split_layers = [num_layers // 2]
                print(f"Set split_layers dynamically to middle layer: {self.split_layers}")
            else:
                raise ValueError(f"Invalid split_layer type: {type(split_layer)}")

        # For backward compatibility, split_layer references the first split layer
        self.split_layer = self.split_layers[0]

        # Robust hidden size detection
        if hasattr(self.model.config, "text_config") and hasattr(self.model.config.text_config, "hidden_size"):
            self.hidden_size = self.model.config.text_config.hidden_size
        elif hasattr(self.model.config, "hidden_size"):
            self.hidden_size = self.model.config.hidden_size
        elif hasattr(self.model.config, "n_embd"):
            self.hidden_size = self.model.config.n_embd
        elif hasattr(self.model.config, "d_model"):
            self.hidden_size = self.model.config.d_model
        else:
            self.hidden_size = 2048
            print(f"Warning: Could not detect hidden_size from model config. Using default: {self.hidden_size}")

        self.predictor_type = resolved_predictor_type

        # Define predictors module dict
        self.predictors = nn.ModuleDict()
        for lyr in self.split_layers:
            lyr_device = next(self.layers[lyr].parameters()).device
            print(f"Split layer {lyr} is on device: {lyr_device}")
            self.predictors[str(lyr)] = self._create_predictor(self.predictor_type, self.hidden_size, lyr_device)

        # Task Conditioning Embedding
        self.num_tasks = num_tasks
        if num_tasks is not None:
            self.task_embeddings = nn.ModuleDict()
            for lyr in self.split_layers:
                lyr_device = next(self.layers[lyr].parameters()).device
                self.task_embeddings[str(lyr)] = nn.Embedding(num_tasks, self.hidden_size).to(lyr_device).to(torch.bfloat16)
            print(f"Initialized learnable task embedding tables for {num_tasks} tasks across split layers.")
        else:
            self.task_embeddings = None

        self.current_task_id = None

        # Register forward hooks
        self.hook_handles = []
        for lyr in self.split_layers:
            # Capturing current layer index with a default argument in the lambda
            hook_fn = lambda m, i, o, l=lyr: self.hook_fn(m, i, o, l)
            handle = self.layers[lyr].register_forward_hook(hook_fn)
            self.hook_handles.append(handle)

        # To keep backward compatibility, self.hook_handle is the first handle
        self.hook_handle = self.hook_handles[0]

        self.use_predictor = False
        self.last_thought_vectors = {}

        if predictor_path and os.path.exists(predictor_path):
            self.load_predictor(predictor_path)

    @property
    def predictor(self):
        """Property returning the module dict of predictors (or single predictor) for backward compatibility."""
        if len(self.split_layers) == 1:
            return self.predictors[str(self.split_layers[0])]
        return self.predictors

    def _create_predictor(self, predictor_type, hidden_size, device):
        if predictor_type == "mlp":
            mlp = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size * 2),
                nn.SiLU(),
                nn.Linear(hidden_size * 2, hidden_size)
            )
            # Initialize final projection layer to zero (weights & biases) to start as identity mapping
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)
            return mlp.to(device).to(torch.bfloat16)
        elif predictor_type in ["transformer", "trs"]:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
            encoder_layer = TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 2,
                dropout=0.0,
                activation='gelu',
                batch_first=True,
                norm_first=True,
                layer_norm_eps=1e-5
            )
            transformer_block = TransformerEncoder(encoder_layer, num_layers=1)
            
            # Zero-initialize attention and FFN output projections to ensure identity start mapping
            for layer in transformer_block.layers:
                if hasattr(layer, "linear2"):
                    nn.init.zeros_(layer.linear2.weight)
                    nn.init.zeros_(layer.linear2.bias)
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "out_proj"):
                    nn.init.zeros_(layer.self_attn.out_proj.weight)
                    nn.init.zeros_(layer.self_attn.out_proj.bias)
            
            class TransformerDeltaPredictor(nn.Module):
                def __init__(self, trs):
                    super().__init__()
                    self.trs = trs
                def forward(self, x):
                    # trs(x) includes standard residual connection within the encoder layers:
                    # x_new = x + Attention(x) + FFN(x).
                    # Since we zero-initialize the output projections of attention and FFN,
                    # trs(x) initially returns exactly x, and the subtraction returns 0.
                    return self.trs(x) - x
                    
            return TransformerDeltaPredictor(transformer_block).to(device).to(torch.bfloat16)
        else:
            raise ValueError(f"Unknown predictor type: {predictor_type}")

    def load_predictor(self, path):
        print(f"Loading predictor weights from {path}...")
        checkpoint = torch.load(path, map_location=next(self.predictors.parameters()).device)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        
        # Check if legacy MLP checkpoint (which lacks LayerNorm at index 0 and Linear at index 3)
        is_legacy_mlp = False
        if self.predictor_type == "mlp":
            has_index_3 = any(k.endswith(".3.weight") or k == "3.weight" for k in state_dict.keys())
            if not has_index_3:
                is_legacy_mlp = True
                print("Detected legacy MLP checkpoint without LayerNorm. Applying index-shifting mapping.")

        # Clean state dict to match current structure (handling prefixes and LayerNorm shift)
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            k_clean = k
            if k_clean.startswith("predictors."):
                k_clean = k_clean[len("predictors."):]
            elif k_clean.startswith("predictor."):
                k_clean = k_clean[len("predictor."):]
                
            if self.predictor_type == "mlp" and is_legacy_mlp:
                parts = k_clean.split(".")
                mapped_parts = []
                for part in parts:
                    if part == "0":
                        mapped_parts.append("1")
                    elif part == "2":
                        mapped_parts.append("3")
                    else:
                        mapped_parts.append(part)
                k_clean = ".".join(mapped_parts)
            cleaned_state_dict[k_clean] = v
            
        # Map to self.predictors keys (prepending split layer index if not present)
        first_layer_str = str(self.split_layers[0])
        final_state_dict = {}
        for k, v in cleaned_state_dict.items():
            first_part = k.split(".")[0]
            if first_part not in [str(lyr) for lyr in self.split_layers]:
                final_state_dict[f"{first_layer_str}.{k}"] = v
            else:
                final_state_dict[k] = v

        # Load with strict=False to allow missing LayerNorm parameters (which remain at default identity initialization)
        missing_keys, unexpected_keys = self.predictors.load_state_dict(final_state_dict, strict=False)
        if missing_keys:
            non_ln_missing = [k for k in missing_keys if not (k.endswith(".0.weight") or k.endswith(".0.bias"))]
            if non_ln_missing:
                print(f"Warning: Missing keys in state dict: {non_ln_missing}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in state dict: {unexpected_keys}")
            
        print("Successfully loaded predictors state dict.")

        if isinstance(checkpoint, dict) and "task_embeddings_state_dict" in checkpoint and self.task_embeddings is not None:
            task_emb_sd = checkpoint["task_embeddings_state_dict"]
            is_legacy_task_emb = "weight" in task_emb_sd
            if is_legacy_task_emb:
                for lyr in self.split_layers:
                    self.task_embeddings[str(lyr)].load_state_dict(task_emb_sd)
                print("Successfully loaded legacy single task embedding table into all layers.")
            else:
                self.task_embeddings.load_state_dict(task_emb_sd)
                print("Successfully loaded layer-specific task embeddings.")
            print("Successfully loaded task embeddings.")

    def save_predictor(self, path):
        print(f"Saving predictor weights to {path}...")
        checkpoint = {
            "predictor_type": self.predictor_type,
            "split_layers": self.split_layers,
            "state_dict": self.predictors.state_dict()
        }
        if self.task_embeddings is not None:
            checkpoint["task_embeddings_state_dict"] = self.task_embeddings.state_dict()
        torch.save(checkpoint, path)

    def hook_fn(self, module, input, output, layer_idx):
        if self.use_predictor:
            is_tuple = isinstance(output, tuple)
            hidden_states = output[0] if is_tuple else output

            # Prefill phase: sequence length > 1
            if hidden_states.shape[1] > 1:
                predictor = self.predictors[str(layer_idx)]
                x = hidden_states
                
                # Apply task conditioning if configured
                if self.task_embeddings is not None:
                    task_id_val = self.current_task_id if self.current_task_id is not None else 0
                    if not isinstance(task_id_val, torch.Tensor):
                        task_id_tensor = torch.tensor([task_id_val], device=hidden_states.device)
                    else:
                        task_id_tensor = task_id_val
                    
                    batch_size = hidden_states.shape[0]
                    if task_id_tensor.ndim == 1 and task_id_tensor.shape[0] == 1:
                        task_id_tensor = task_id_tensor.expand(batch_size)
                        
                    task_emb_layer = self.task_embeddings[str(layer_idx)]
                    task_emb = task_emb_layer(task_id_tensor).unsqueeze(1) # [batch_size, 1, hidden_size]
                    x = x + task_emb
                
                predictor_delta = predictor(x)
                
                # Save thought vector for interpretability
                self.last_thought_vectors[str(layer_idx)] = predictor_delta.detach().cpu()
                
                transformed_states = hidden_states + predictor_delta
                
                # Disable predictor intervention after the last split layer has run in this forward pass
                if layer_idx == max(self.split_layers):
                    self.use_predictor = False

                if is_tuple:
                    return (transformed_states,) + output[1:]
                else:
                    return transformed_states
        return output

    def generate_text(self, prompt, max_new_tokens=50, use_predictor=True, task_id=None, do_sample=False, **generation_kwargs):
        self.use_predictor = use_predictor
        self.current_task_id = task_id
        self.last_thought_vectors = {}
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    **generation_kwargs
                )
            generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
            return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        finally:
            self.current_task_id = None

    def sample(self, prompt_ids, max_new_tokens=50, temperature=1.0, top_k=50, top_p=1.0, use_predictor=True, task_id=None):
        """Samples tokens autoregressively from the model using the current predictor policy.
        
        Uses KV caching to speed up token generation.
        
        Returns:
            gen_tokens (Tensor): The token IDs generated, shape [gen_len].
            gen_log_probs (Tensor): Log probabilities of each generated token, shape [gen_len].
        """
        self.use_predictor = use_predictor
        self.current_task_id = task_id
        self.last_thought_vectors = {}
        
        try:
            device = prompt_ids.device
            input_ids = prompt_ids.clone()
            
            gen_tokens = []
            gen_log_probs = []
            
            past_key_values = None
            
            for step in range(max_new_tokens):
                if past_key_values is not None:
                    model_inputs = {
                        "input_ids": input_ids[:, -1:],
                        "use_cache": True,
                        "past_key_values": past_key_values
                    }
                else:
                    model_inputs = {
                        "input_ids": input_ids,
                        "use_cache": True,
                        "past_key_values": None
                    }
                
                with torch.no_grad():
                    outputs = self.model(**model_inputs)
                
                logits = outputs.logits[:, -1, :] # shape [batch_size, vocab_size]
                past_key_values = outputs.past_key_values
                
                # Apply temperature
                if temperature != 1.0 and temperature > 0:
                    logits = logits / temperature
                
                # Apply top-k
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits = logits.masked_fill(indices_to_remove, -float("Inf"))
                
                # Apply top-p (nucleus sampling)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                    logits = logits.masked_fill(indices_to_remove, -float("Inf"))
                
                probs = torch.softmax(logits, dim=-1)
                
                if temperature == 0:
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)
                else:
                    next_token = torch.multinomial(probs, num_samples=1)
                
                log_probs = torch.log_softmax(logits, dim=-1)
                next_token_log_prob = log_probs.gather(dim=-1, index=next_token)
                
                gen_tokens.append(next_token.squeeze(-1))
                gen_log_probs.append(next_token_log_prob.squeeze(-1))
                
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
            
            if len(gen_tokens) == 0:
                return torch.tensor([], dtype=torch.long, device=device), torch.tensor([], device=device)
                
            return torch.cat(gen_tokens, dim=-1), torch.cat(gen_log_probs, dim=-1)
            
        finally:
            self.current_task_id = None

    def compute_sequence_log_probs(self, prompt_ids, gen_tokens, use_predictor=True, task_id=None):
        """Computes the log probabilities of gen_tokens given prompt_ids.
        
        Gradients can flow through the predictor parameters.
        
        Returns:
            sequence_log_prob (Tensor): Scalar tensor of the sum of log probabilities.
        """
        self.use_predictor = use_predictor
        self.current_task_id = task_id
        
        try:
            prompt_len = prompt_ids.shape[-1]
            full_ids = torch.cat([prompt_ids, gen_tokens.unsqueeze(0)], dim=-1)
            attention_mask = torch.ones_like(full_ids)
            
            outputs = self.model(
                input_ids=full_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # Get logits for the generated tokens
            logits = outputs.logits[:, prompt_len - 1 : -1, :] # shape: [1, G, vocab_size]
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Gather log probs of the actual generated tokens
            gen_log_probs = log_probs.gather(dim=-1, index=gen_tokens.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
            return gen_log_probs.sum(dim=-1)
        finally:
            self.current_task_id = None

    def extract_thought_vectors(self, prompt, task_id=None):
        """Extracts the thought vectors (predictor outputs) for each split layer on the given prompt.
        
        Returns:
            dict: Mapping layer index (str) to the thought vector tensor of shape [batch_size, seq_len, hidden_size]
        """
        self.use_predictor = True
        self.current_task_id = task_id
        self.last_thought_vectors = {}
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            self.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            
        return self.last_thought_vectors

    def analyze_thought_similarity(self, prompt, target_cot_and_solution, task_id=None):
        """Measures the cosine similarity of hidden states after injection vs. the ground-truth CoT states.
        
        Returns:
            dict: Mapping layer index (str) to cosine similarity scores.
        """
        inputs_prompt = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs_prompt.input_ids.shape[-1]
        
        target_text = f"<think>\n{target_cot_and_solution}" if not target_cot_and_solution.startswith("<think>") else target_cot_and_solution
        inputs_full = self.tokenizer(prompt + target_text, return_tensors="pt").to(self.model.device)
        
        self.use_predictor = False
        with torch.no_grad():
            outputs_full = self.model(
                input_ids=inputs_full.input_ids,
                attention_mask=inputs_full.attention_mask,
                output_hidden_states=True
            )
            
        self.use_predictor = True
        self.current_task_id = task_id
        with torch.no_grad():
            outputs_prompt = self.model(
                input_ids=inputs_prompt.input_ids,
                attention_mask=inputs_prompt.attention_mask,
                output_hidden_states=True
            )
            
        similarities = {}
        for lyr in self.split_layers:
            target_latents = outputs_full.hidden_states[lyr + 1][:, prompt_len:, :]
            target_thought = target_latents.mean(dim=1) # [1, hidden_size]
            
            predicted_latents = outputs_prompt.hidden_states[lyr + 1]
            predicted_thought = predicted_latents[:, -1, :] # [1, hidden_size]
            
            cos_sim = torch.nn.functional.cosine_similarity(predicted_thought, target_thought, dim=-1).item()
            similarities[str(lyr)] = cos_sim
            
        return similarities

    def analyze_layer_trajectory_similarity(self, prompt, target_cot_and_solution, task_id=None):
        """Compares layer-by-layer trajectory of hidden states after injection vs. full CoT path.
        
        Returns:
            dict: Layer index (int) -> cosine similarity between injected states and target CoT states.
        """
        inputs_prompt = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs_prompt.input_ids.shape[-1]
        
        target_text = f"<think>\n{target_cot_and_solution}" if not target_cot_and_solution.startswith("<think>") else target_cot_and_solution
        inputs_full = self.tokenizer(prompt + target_text, return_tensors="pt").to(self.model.device)
        
        self.use_predictor = False
        with torch.no_grad():
            outputs_full = self.model(
                input_ids=inputs_full.input_ids,
                attention_mask=inputs_full.attention_mask,
                output_hidden_states=True
            )
            
        self.use_predictor = True
        self.current_task_id = task_id
        with torch.no_grad():
            outputs_prompt = self.model(
                input_ids=inputs_prompt.input_ids,
                attention_mask=inputs_prompt.attention_mask,
                output_hidden_states=True
            )
            
        trajectory_similarities = {}
        # Iterate over all layers (0 to num_layers)
        num_layers = len(outputs_full.hidden_states)
        for lyr_idx in range(num_layers):
            # target trajectory average representation over the CoT tokens
            target_rep = outputs_full.hidden_states[lyr_idx][:, prompt_len:, :].mean(dim=1)
            # injected trajectory representation at the last prompt token
            injected_rep = outputs_prompt.hidden_states[lyr_idx][:, -1, :]
            
            cos_sim = torch.nn.functional.cosine_similarity(injected_rep, target_rep, dim=-1).item()
            trajectory_similarities[lyr_idx] = cos_sim
            
        return trajectory_similarities

    def measure_logit_bias(self, prompt, task_id=None):
        """Measures how much the thought vector biases the output distribution.
        
        Returns:
            dict: Metrics comparing logits/probabilities with and without injection.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # 1. Without injection
        self.use_predictor = False
        with torch.no_grad():
            outputs_off = self.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            logits_off = outputs_off.logits[:, -1, :] # [batch_size, vocab_size]
            probs_off = torch.softmax(logits_off, dim=-1)
            
        # 2. With injection
        self.use_predictor = True
        self.current_task_id = task_id
        with torch.no_grad():
            outputs_on = self.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            logits_on = outputs_on.logits[:, -1, :]
            probs_on = torch.softmax(logits_on, dim=-1)
            
        # Compute divergence/similarity metrics
        kl_div = torch.sum(probs_off * (torch.log(probs_off + 1e-10) - torch.log(probs_on + 1e-10)), dim=-1).mean().item()
        cos_sim_logits = torch.nn.functional.cosine_similarity(logits_off, logits_on, dim=-1).mean().item()
        
        # Get top-5 predictions for qualitative check
        top_k = 5
        top_probs_off, top_ids_off = torch.topk(probs_off[0], k=top_k)
        top_probs_on, top_ids_on = torch.topk(probs_on[0], k=top_k)
        
        tokens_off = [self.tokenizer.decode([i]) for i in top_ids_off]
        tokens_on = [self.tokenizer.decode([i]) for i in top_ids_on]
        
        return {
            "kl_divergence": kl_div,
            "cosine_similarity_logits": cos_sim_logits,
            "top_tokens_off": list(zip(tokens_off, top_probs_off.tolist())),
            "top_tokens_on": list(zip(tokens_on, top_probs_on.tolist()))
        }

def project_thoughts_pca(thought_vectors, n_components=2):
    """Projects a list of thought vectors to a lower-dimensional space using PCA SVD.
    
    Returns:
        numpy.ndarray: Projected coordinates of shape [N, n_components].
    """
    if isinstance(thought_vectors, list):
        stacked = torch.stack([t.detach().cpu().squeeze() for t in thought_vectors], dim=0)
    else:
        stacked = thought_vectors.detach().cpu().squeeze()
        
    if stacked.ndim == 1:
        stacked = stacked.unsqueeze(0)
        
    # Convert to float32 to avoid dtype mismatches and support SVD operations
    stacked = stacked.to(torch.float32)
    
    mean = torch.mean(stacked, dim=0, keepdim=True)
    centered = stacked - mean
    
    # pca_lowrank requires the dimension size to be at least q
    q = min(n_components, centered.shape[0])
    
    try:
        U, S, V = torch.pca_lowrank(centered, q=q)
        projected = torch.matmul(centered, V[:, :q])
    except Exception:
        # Fallback to standard SVD if pca_lowrank fails
        U, S, V = torch.linalg.svd(centered, full_matrices=False)
        projected = torch.matmul(centered, V.mH[:, :q])
        
    return projected.numpy()
