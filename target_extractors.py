import torch

class ResponseTargetExtractor:
    """Standard target extractor that targets the mean representation of the generated response."""
    def __call__(self, tokenizer, full_outputs, full_input_ids, prompt_len, split_layers):
        target_thoughts = {}
        for lyr in split_layers:
            target_latents = full_outputs.hidden_states[lyr + 1][:, prompt_len:, :]
            target_thoughts[str(lyr)] = target_latents.mean(dim=1)
        return target_thoughts

    def get_trajectory_rep(self, tokenizer, lyr_hidden_state, full_input_ids, prompt_len):
        return lyr_hidden_state[:, prompt_len:, :].mean(dim=1)


class ThinkingTargetExtractor:
    """Reasoning-focused target extractor that targets the last token representation of the thinking block (CoT)."""
    def __call__(self, tokenizer, full_outputs, full_input_ids, prompt_len, split_layers):
        # Search for </think> token ID
        end_think_ids = tokenizer("</think>", add_special_tokens=False).input_ids
        seq = full_input_ids[0].tolist()
        sub_len = len(end_think_ids)
        thinking_end_idx = None
        if sub_len > 0:
            for i in range(len(seq) - sub_len + 1):
                if seq[i : i + sub_len] == end_think_ids:
                    thinking_end_idx = i + sub_len - 1
                    break
        
        if thinking_end_idx is None:
            # Fallback to the last token of the prompt if </think> is not found
            thinking_end_idx = prompt_len - 1
            
        target_thoughts = {}
        for lyr in split_layers:
            target_latents = full_outputs.hidden_states[lyr + 1]
            target_thoughts[str(lyr)] = target_latents[:, thinking_end_idx, :]
        return target_thoughts

    def get_trajectory_rep(self, tokenizer, lyr_hidden_state, full_input_ids, prompt_len):
        end_think_ids = tokenizer("</think>", add_special_tokens=False).input_ids
        seq = full_input_ids[0].tolist()
        sub_len = len(end_think_ids)
        thinking_end_idx = None
        if sub_len > 0:
            for i in range(len(seq) - sub_len + 1):
                if seq[i : i + sub_len] == end_think_ids:
                    thinking_end_idx = i + sub_len - 1
                    break
        if thinking_end_idx is None:
            thinking_end_idx = prompt_len - 1
        return lyr_hidden_state[:, thinking_end_idx, :]
