import datetime
import os
import sys
from mimetypes import init

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from fastcore.all import *

""" All-Reduce example."""
device = torch.device("cpu")


def all_reduce_example(rank):
    """Simple collective communication."""
    group = dist.new_group([0, 1])
    tensor = torch.ones(1).to(device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print("Rank ", rank, " has data ", tensor[0])


@call_parse
def main(
    rank: int,  # node's position in 0...world_size-1
    world_size: int = 2,
    master_addr: str = "della-vis1",
):
    assert torch.distributed.is_available()
    print(f"Initializing process group with rank {rank}")
    torch.distributed.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://{master_addr}:31343",
        timeout=datetime.timedelta(seconds=60),
    )
    print(f"pytorch device {device}, {torch.cuda.is_available()}")
    print(f"Running all_reduce example")
    all_reduce_example(rank)
