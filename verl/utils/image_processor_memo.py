import hashlib
import os
from collections import OrderedDict

from PIL import Image


_DIGESTS = OrderedDict()


def _image_key(image):
    if not isinstance(image, Image.Image):
        return None
    known = _DIGESTS.get(id(image))
    if known is not None and known[0] is image:
        return known[1]
    key = (image.mode, image.size, hashlib.sha256(image.tobytes()).digest())
    _DIGESTS[id(image)] = (image, key)
    while len(_DIGESTS) > 64:
        _DIGESTS.popitem(last=False)
    return key


class MemoImageProcessor:
    def __init__(self, inner, max_entries=32):
        self._inner = inner
        self._cache = OrderedDict()
        self._max_entries = max_entries
        self.hits = self.misses = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __call__(self, images=None, **kwargs):
        listed = images if isinstance(images, (list, tuple)) else [images]
        keys = [_image_key(image) for image in listed]
        if images is None or any(key is None for key in keys):
            return self._inner(images=images, **kwargs)
        key = (isinstance(images, (list, tuple)), tuple(keys), repr(sorted(kwargs.items())))
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            self.hits += 1
            return cached.__class__(dict(cached), tensor_type=None)
        self.misses += 1
        result = self._inner(images=images, **kwargs)
        self._cache[key] = result
        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)
        return result.__class__(dict(result), tensor_type=None)


def memoize_image_processor(processor):
    if processor is None or os.environ.get("VERL_MEMO_IMAGE_PROCESSOR", "1") != "1":
        return processor
    inner = getattr(processor, "image_processor", None)
    if inner is None or isinstance(inner, MemoImageProcessor):
        return processor
    processor.image_processor = MemoImageProcessor(inner)
    return processor


def _source_key(item):
    image = item.get("image", item.get("image_url"))
    if isinstance(image, dict):
        image = image.get("path") or image.get("bytes") or image.get("url")
    if isinstance(image, str) and not image.startswith(("http://", "https://", "data:")):
        path = image[7:] if image.startswith("file://") else image
        try:
            stat = os.stat(path)
        except OSError:
            return None
        return ("file", os.path.realpath(path), stat.st_size, stat.st_mtime_ns)
    if isinstance(image, (bytes, bytearray)):
        return ("bytes", hashlib.sha256(image).digest())
    if isinstance(image, Image.Image):
        return ("pil", _image_key(image))
    return None


class VisionInfoMemo:

    def __init__(self, max_entries=32):
        self._cache = OrderedDict()
        self._max_entries = max_entries
        self.hits = self.misses = 0

    def key(self, messages):
        parts = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") not in ("image", "image_url", "video"):
                    continue
                if item.get("type") == "video":
                    return None
                key = _source_key(item)
                if key is None:
                    return None
                extras = tuple(sorted((k, repr(v)) for k, v in item.items() if k not in ("image", "image_url", "type")))
                parts.append((key, extras))
        return tuple(parts) if parts else None

    def get(self, key):
        value = self._cache.get(key) if key is not None else None
        if value is not None:
            self._cache.move_to_end(key)
            self.hits += 1
        return value

    def put(self, key, value):
        if key is None:
            return
        self.misses += 1
        self._cache[key] = value
        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)
