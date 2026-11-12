import hashlib

import numpy as np
from PIL import Image
import torch


def _content_key(value):
    if isinstance(value, Image.Image):
        return ("pil", value.mode, value.size, hashlib.sha256(value.tobytes()).digest())
    if isinstance(value, torch.Tensor):
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        return ("tensor", str(value.dtype), tuple(value.shape), hashlib.sha256(raw.tobytes()).digest())
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return ("objects", tuple(value.shape), _content_key(value.tolist()))
        return ("ndarray", str(value.dtype), tuple(value.shape), hashlib.sha256(value.tobytes()).digest())
    if isinstance(value, dict):
        return ("dict", tuple((key, _content_key(item)) for key, item in sorted(value.items())))
    if isinstance(value, (list, tuple)):
        return ("sequence", tuple(_content_key(item) for item in value))
    if isinstance(value, (str, int, float, bool, bytes, type(None))):
        return (type(value).__name__, value)
    raise TypeError(f"Uncacheable teacher message component: {type(value).__name__}")


class TeacherPrefixCache:
    def __init__(self, max_entries=512):
        self.entries = {}
        self.max_entries = max_entries
        self.hits = self.misses = 0

    def get_or_build(self, messages, max_prompt_len, builder):
        try:
            key = (max_prompt_len, _content_key(messages))
        except TypeError:
            self.misses += 1
            return builder()
        if key in self.entries:
            self.hits += 1
            return self.entries[key]
        self.misses += 1
        value = builder()
        if len(self.entries) < self.max_entries:
            self.entries[key] = value
        return value
