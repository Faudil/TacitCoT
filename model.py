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
    def __init__(self, model_name: str, split_layer=18, predictor_path=None, predictor_type="mlp"):
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

        if split_layer is None or split_layer >= num_layers:
            self.split_layer = num_layers // 2
            print(f"Set split_layer dynamically to middle layer: {self.split_layer}")
        else:
            self.split_layer = split_layer

        if hasattr(self.model.config, "text_config"):
            self.hidden_size = self.model.config.text_config.hidden_size
        else:
            self.hidden_size = getattr(self.model.config, "hidden_size", 2048)

        split_layer_device = next(self.layers[self.split_layer].parameters()).device
        print(f"Split layer {self.split_layer} is on device: {split_layer_device}")

        # Resolve predictor_type from checkpoint if available
        resolved_predictor_type = predictor_type
        if predictor_path and os.path.exists(predictor_path):
            try:
                checkpoint = torch.load(predictor_path, map_location="cpu")
                if isinstance(checkpoint, dict) and "predictor_type" in checkpoint:
                    resolved_predictor_type = checkpoint["predictor_type"]
                    print(f"Detected predictor type '{resolved_predictor_type}' from checkpoint metadata.")
            except Exception as e:
                print(f"Note: Could not inspect checkpoint metadata: {e}")

        self.predictor_type = resolved_predictor_type

        if self.predictor_type == "mlp":
            self.predictor = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size * 2),
                nn.SiLU(),
                nn.Linear(self.hidden_size * 2, self.hidden_size)
            ).to(split_layer_device).to(torch.bfloat16)
        elif self.predictor_type in ["transformer", "trs"]:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
            encoder_layer = TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=8,
                dim_feedforward=self.hidden_size * 2,
                dropout=0.0,
                activation='gelu',
                batch_first=True,
                norm_first=True,
                layer_norm_eps=1e-5
            )
            transformer_block = TransformerEncoder(encoder_layer, num_layers=1)
            
            class TransformerDeltaPredictor(nn.Module):
                def __init__(self, trs):
                    super().__init__()
                    self.trs = trs
                def forward(self, x):
                    return self.trs(x) - x
                    
            self.predictor = TransformerDeltaPredictor(transformer_block).to(split_layer_device).to(torch.bfloat16)
        else:
            raise ValueError(f"Unknown predictor type: {self.predictor_type}")

        self.hook_handle = self.layers[self.split_layer].register_forward_hook(self.hook_fn)
        self.use_predictor = False

        if predictor_path and os.path.exists(predictor_path):
            self.load_predictor(predictor_path)

    def load_predictor(self, path):
        print(f"Loading predictor weights from {path}...")
        try:
            checkpoint = torch.load(path, map_location=next(self.predictor.parameters()).device)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                self.predictor.load_state_dict(checkpoint["state_dict"])
            else:
                self.predictor.load_state_dict(checkpoint)
        except Exception as e:
            print(f"WARNING: Could not load predictor weights from {path} due to size mismatch or error: {e}")
            print("Initializing predictor weights from scratch.")

    def save_predictor(self, path):
        print(f"Saving predictor weights to {path}...")
        checkpoint = {
            "predictor_type": self.predictor_type,
            "state_dict": self.predictor.state_dict()
        }
        torch.save(checkpoint, path)

    def hook_fn(self, module, input, output):
        if self.use_predictor:
            is_tuple = isinstance(output, tuple)
            hidden_states = output[0] if is_tuple else output

            if hidden_states.shape[1] > 1:
                transformed_states = hidden_states + self.predictor(hidden_states)
                if is_tuple:
                    return (transformed_states,) + output[1:]
                else:
                    return transformed_states
        return output

    def generate_text(self, prompt, max_new_tokens=50, use_predictor=True):
        self.use_predictor = use_predictor
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
