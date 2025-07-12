import math
import torch

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    # Ensure query, key, value are on the same device
    device = query.device
    query, key, value = query.to(device), key.to(device), value.to(device)

    # Dimensions
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

    # Pre-compute causal mask if required
    if is_causal:
        assert attn_mask is None  # Avoid conflicts between causal mask and attn_mask
        causal_mask = torch.tril(torch.ones(L, S, dtype=torch.bool, device=device))
    else:
        causal_mask = None

    # Initialize attention bias directly on device
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=device)
    if is_causal and causal_mask is not None:
        attn_bias.masked_fill_(causal_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask.to(device)

    # Compute attention weights
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)

    # Dropout (use training mode only if applicable)
    if dropout_p > 0.0:
        attn_weight = torch.dropout(attn_weight, dropout_p, train=query.requires_grad)

    # Compute final attention output
    return attn_weight @ value, attn_weight