import torch


def cut_reference_row(input_ids, attention_mask, position_ids, response_start_idx, image_token_id):
    if position_ids.dim() != 2 or position_ids.shape[0] != 4:
        raise ValueError(f"cut reference needs four-row Qwen3.5 position ids, got {tuple(position_ids.shape)}")
    start = int(response_start_idx)
    keep = input_ids != image_token_id
    if not bool(keep[start:].all()):
        raise ValueError("image placeholder tokens inside the response")
    n_removed = int((~keep[:start]).sum())
    ids = input_ids[keep]
    mask = attention_mask[keep]
    pos = position_ids[:, keep].clone()
    pos[0] = torch.arange(ids.shape[0], dtype=pos.dtype, device=pos.device)
    return ids, mask, pos, torch.tensor(start - n_removed, dtype=torch.long), n_removed
