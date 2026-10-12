"""torchrun helpers. Works the same with one process (python -m ...) and on CPU (gloo backend)."""
import datetime
import os

import torch
import torch.distributed as dist


def setup():
    """Returns (rank, world_size, device). Initializes the process group when launched by torchrun."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo",
                                timeout=datetime.timedelta(minutes=60))
    return rank, world, device


def barrier():
    if dist.is_initialized():
        dist.barrier()


def gather(items):
    """Concatenate per-rank lists on every rank."""
    if not dist.is_initialized():
        return items
    parts = [None] * dist.get_world_size()
    dist.all_gather_object(parts, items)
    return [x for part in parts for x in part]


def all_sum(value, device):
    if not dist.is_initialized():
        return value
    t = torch.tensor([float(value)], device=device)
    dist.all_reduce(t)
    return t.item()


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()
