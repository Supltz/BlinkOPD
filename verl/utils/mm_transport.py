from dataclasses import dataclass
import warnings

import numpy as np
import torch


@dataclass(frozen=True)
class TensorBytes:

    data: np.ndarray
    dtype: str
    shape: tuple


def multi_modal_inputs_to_numpy(rows):
    memo = {}

    def encode(value):
        if isinstance(value, torch.Tensor):
            key = id(value)
            if key not in memo:
                cpu = value.detach().cpu()
                try:
                    packed = cpu.numpy()
                except TypeError:
                    raw = cpu.contiguous().reshape(-1).view(torch.uint8).numpy()
                    packed = TensorBytes(raw, str(cpu.dtype).split(".")[-1], tuple(cpu.shape))
                memo[key] = (value, packed)
            return memo[key][1]
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [encode(item) for item in value]
        if isinstance(value, tuple):
            return tuple(encode(item) for item in value)
        return value

    out = np.empty(len(rows), dtype=object)
    out[:] = [encode(row) for row in rows]
    return out


def multi_modal_tensor(value, *, writable=False):
    if isinstance(value, TensorBytes):
        dtype = getattr(torch, value.dtype)
        return multi_modal_tensor(value.data, writable=writable).view(dtype).reshape(value.shape)
    if not isinstance(value, np.ndarray):
        return value
    if writable and not value.flags.writeable:
        value = value.copy()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
        return torch.from_numpy(value)


def prepare_actor_transport(batch):
    keys = [key for key in batch.non_tensor_batch
            if key == "uid" or key == "multi_modal_inputs" or key.endswith("_multi_modal_inputs")]
    result = batch.select(non_tensor_batch_keys=keys)
    for key in keys:
        if key != "uid":
            result.non_tensor_batch[key] = multi_modal_inputs_to_numpy(result.non_tensor_batch[key])
    return result
