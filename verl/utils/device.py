

import logging

import torch

logger = logging.getLogger(__name__)


def is_torch_npu_available(check_device=True) -> bool:
    pass
    try:
        if not hasattr(torch, "npu"):
            return False

        if check_device:
            return torch.npu.is_available()
        else:
            return True
    except ImportError:
        return False


is_cuda_available = torch.cuda.is_available()
is_npu_available = is_torch_npu_available()


def get_visible_devices_keyword() -> str:
    pass
    return "CUDA_VISIBLE_DEVICES" if not is_torch_npu_available(check_device=False) else "ASCEND_RT_VISIBLE_DEVICES"


def get_device_name() -> str:
    pass
    if is_cuda_available:
        device = "cuda"
    elif is_npu_available:
        device = "npu"
    else:
        device = "cpu"
    return device


def get_torch_device():
    pass
    device_name = get_device_name()
    try:
        return getattr(torch, device_name)
    except AttributeError:
        logger.warning(f"Device namespace '{device_name}' not found in torch, try to load torch.cuda.")
        return torch.cuda


def get_device_id() -> int:
    pass
    return get_torch_device().current_device()


def get_nccl_backend() -> str:
    pass
    if is_npu_available:
        return "hccl"
    else:

        return "nccl"


def set_expandable_segments(enable: bool) -> None:
    pass
    if is_cuda_available:
        torch.cuda.memory._set_allocator_settings(f"expandable_segments:{enable}")


def auto_set_device(config) -> None:
    pass
    if config and hasattr(config, "trainer") and hasattr(config.trainer, "device"):
        if is_torch_npu_available():
            if config.trainer.device not in ["cpu", "npu"]:
                logger.warning(
                    f"Detect setting config.trainer.device to {config.trainer.device} for Ascend NPU, maybe"
                    f"from default value in config file, automatically set to `npu` instead."
                )

            config.trainer.device = "npu"


def get_device_capability(device_id: int = 0) -> tuple[int | None, int | None]:
    pass
    major, minor = None, None
    if is_cuda_available:
        major, minor = torch.cuda.get_device_capability(device_id)

    return major, minor
