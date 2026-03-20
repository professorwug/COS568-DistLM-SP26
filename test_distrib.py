import os
import sys
from mimetypes import init

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from fastcore.all import *

""" All-Reduce example."""


def all_reduce_example(rank, size):
    """Simple collective communication."""
    group = dist.new_group([0, 1, 2, 3])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print("Rank ", rank, " has data ", tensor[0])


@call_parse
def main(
    rank: int,  # node's position in 0...world_size-1
    world_size: int = 4,
    master_addr: str = "della-vis1.princeton.edu",
):
    assert torch.distributed.is_available()
    torch.distributed.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://{master_addr}:13471",
    )
    all_reduce_example(rank)
