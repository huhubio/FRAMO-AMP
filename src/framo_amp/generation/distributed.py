"""Small distributed helpers shared by generation and LoRA training."""
from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    """Return True when the process was launched by torchrun."""
    return int(os.environ.get("LOCAL_RANK", "-1")) != -1


def init_distributed_if_needed() -> None:
    """Initialize torch.distributed when running under torchrun."""
    if is_distributed() and dist.is_available() and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_rank_zero() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_gather_object(obj: Any) -> list[Any]:
    if not (dist.is_available() and dist.is_initialized()):
        return [obj]
    gathered: list[Any] = [None for _ in range(get_world_size())]
    dist.all_gather_object(gathered, obj)
    return gathered


def unwrap_model(model):  # noqa: ANN001, ANN201
    """Return the underlying module when a model is wrapped by DDP/DataParallel."""
    return model.module if hasattr(model, "module") else model
