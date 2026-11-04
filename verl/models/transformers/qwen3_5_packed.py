from functools import lru_cache, wraps
import inspect

import torch
import torch.nn.functional as F


def packed_metadata(input_ids, position_ids, attention_mask=None, past_key_values=None):
    if attention_mask is not None or past_key_values is not None or input_ids.shape[0] != 1:
        return None
    if position_ids is None:
        raise ValueError("Qwen3.5 packed training requires explicit text position_ids")
    if position_ids.ndim == 3:
        if position_ids.shape[0] != 4:
            raise ValueError("Qwen3.5 packed multimodal positions must include the fourth/text channel")
        positions = position_ids[0]
    else:
        positions = position_ids
    if positions.shape != input_ids.shape:
        raise ValueError("Qwen3.5 packed text positions do not match input shape")
    starts = (positions[0] == 0).nonzero(as_tuple=True)[0]
    boundaries = starts.tolist() + [input_ids.shape[1]]
    if boundaries[0] != 0 or len(boundaries) < 2:
        raise ValueError("Qwen3.5 packed sequences must start at text position zero")
    if len(boundaries) == 2:
        return None
    cu = torch.tensor(boundaries, dtype=torch.int32, device=input_ids.device)
    seq_idx = (positions == 0).to(torch.int32).cumsum(-1, dtype=torch.int32) - 1
    return (tuple(boundaries), cu, seq_idx.contiguous())


@lru_cache(maxsize=8)
def _supports_varlen(fn):
    return "cu_seqlens" in inspect.signature(fn).parameters


def _packed_delta_forward(self, hidden_states, metadata):
    boundaries, cu, seq_idx = metadata
    batch_size, seq_len, _ = hidden_states.shape
    if batch_size != 1 or boundaries[-1] != seq_len:
        raise ValueError("Qwen3.5 packed boundary/hidden-state shape mismatch")
    mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
    z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
    b, a = self.in_proj_b(hidden_states), self.in_proj_a(hidden_states)
    if self.causal_conv1d_fn is not None:
        mixed_qkv = self.causal_conv1d_fn(
            x=mixed_qkv, weight=self.conv1d.weight.squeeze(1), bias=self.conv1d.bias,
            activation=self.activation, seq_idx=seq_idx,
        )
    else:
        mixed_qkv = torch.cat([
            F.silu(self.conv1d(mixed_qkv[:, :, lo:hi])[:, :, :hi - lo])
            for lo, hi in zip(boundaries[:-1], boundaries[1:])
        ], dim=2)
    query, key, value = torch.split(
        mixed_qkv.transpose(1, 2), [self.key_dim, self.key_dim, self.value_dim], dim=-1,
    )
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
    opts = dict(initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
    if _supports_varlen(self.chunk_gated_delta_rule):
        core, _ = self.chunk_gated_delta_rule(query, key, value, g=g, beta=beta, cu_seqlens=cu, **opts)
    else:
        core = torch.cat([
            self.chunk_gated_delta_rule(
                query[:, lo:hi], key[:, lo:hi], value[:, lo:hi],
                g=g[:, lo:hi], beta=beta[:, lo:hi], **opts,
            )[0] for lo, hi in zip(boundaries[:-1], boundaries[1:])
        ], dim=1)
    core = self.norm(core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
    return self.out_proj(core.reshape(batch_size, seq_len, -1))


def install_packed_linear_attention():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5GatedDeltaNet

    if getattr(Qwen3_5DecoderLayer, "_verl_packed_linear_attention", False):
        return
    delta_forward = Qwen3_5GatedDeltaNet.forward
    decoder_forward = Qwen3_5DecoderLayer.forward

    @wraps(delta_forward)
    def delta(self, hidden_states, cache_params=None, attention_mask=None, packed_linear_metadata=None):
        if packed_linear_metadata is None:
            return delta_forward(self, hidden_states, cache_params=cache_params, attention_mask=attention_mask)
        if cache_params is not None or attention_mask is not None:
            raise ValueError("Packed linear attention cannot mix generation cache or padding mask")
        return _packed_delta_forward(self, hidden_states, packed_linear_metadata)

    @wraps(decoder_forward)
    def decoder(self, hidden_states, position_embeddings, attention_mask=None, position_ids=None,
                past_key_values=None, packed_linear_metadata=None, **kwargs):
        if packed_linear_metadata is None or self.layer_type != "linear_attention":
            return decoder_forward(
                self, hidden_states, position_embeddings, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values, **kwargs,
            )
        residual = hidden_states
        hidden_states = self.linear_attn(
            hidden_states=self.input_layernorm(hidden_states), cache_params=past_key_values,
            attention_mask=attention_mask, packed_linear_metadata=packed_linear_metadata,
        )
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

    Qwen3_5GatedDeltaNet.forward = delta
    Qwen3_5DecoderLayer.forward = decoder
    Qwen3_5DecoderLayer._verl_packed_linear_attention = True
