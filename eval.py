import torch
import torch.nn as nn

def compute_loss(pred, target, loss_type="mse"):
    """Computes the loss between predicted and target representations.
    
    Supported loss types:
    - 'mse': Mean Squared Error
    - 'cosine': 1 - Cosine Similarity
    - 'combined': MSE + (1 - Cosine Similarity)
    """
    if loss_type == "mse":
        return nn.MSELoss()(pred, target)
    elif loss_type == "cosine":
        cos_sim = torch.nn.functional.cosine_similarity(pred, target, dim=-1).mean()
        return 1.0 - cos_sim
    elif loss_type == "combined":
        mse = nn.MSELoss()(pred, target)
        cos_sim = torch.nn.functional.cosine_similarity(pred, target, dim=-1).mean()
        return mse + (1.0 - cos_sim)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

def val_jepa_step(sandwich_model, prompt_ids, target_ids, loss_type="mse", task_id=None):
    """Computes validation loss between predicted thoughts and targets."""
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
        
        sandwich_model.use_predictor = True
        prompt_attention_mask = torch.ones_like(prompt_ids)
        prompt_outputs = sandwich_model.model(
            input_ids=prompt_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True
        )
        
        total_loss = 0.0
        for lyr in sandwich_model.split_layers:
            predicted_latents = prompt_outputs.hidden_states[lyr + 1]
            predicted_thought = predicted_latents[:, -1, :]
            
            loss = compute_loss(predicted_thought, target_thoughts[str(lyr)], loss_type)
            total_loss = total_loss + loss
            
    sandwich_model.current_task_id = None
    return total_loss.item()
