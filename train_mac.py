# /// script
# dependencies = [
#     "accelerate",
#     "adam-atan2-pytorch>=0.1.18",
#     "setuptools",
#     "titans-pytorch",
#     "tqdm",
#     "wandb"
# ]
# ///

import gzip
import os
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm.auto import tqdm
from adam_atan2_pytorch import AdoptAtan2
from safetensors.torch import save_model
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from rich.progress import Progress

from titans_pytorch import MemoryAsContextTransformer, MemoryAttention, MemoryMLP

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
VALIDATE_EVERY = 100
GENERATE_EVERY = 500
SAVE_EVERY = 100 * VALIDATE_EVERY
PRIME_LENGTH = 100
GENERATE_LENGTH = 512
SHOULD_GENERATE = True
SEQ_LEN = 512

# neural memory related

NEURAL_MEMORY_DEPTH = 2
NUM_PERSIST_MEM = 4
NUM_LONGTERM_MEM = 4
NEURAL_MEM_LAYERS = (2, 4, 6)  # layers 2, 4, 6 have neural memory, can add more
NEURAL_MEM_GATE_ATTN_OUTPUT = False
NEURAL_MEM_MOMENTUM = True
NEURAL_MEM_MOMENTUM_ORDER = 1
NEURAL_MEM_QK_NORM = True
NEURAL_MEM_MAX_LR = 1e-1
USE_MEM_ATTENTION_MODEL = False
WINDOW_SIZE = 32
NEURAL_MEM_SEGMENT_LEN = (
    4  # set smaller for more granularity for learning rate / momentum etc
)
NEURAL_MEM_BATCH_SIZE = 128  # set smaller to update the neural memory weights more often as it traverses the sequence
SLIDING_WINDOWS = True
STORE_ATTN_POOL_CHUNKS = True  # whether to use attention pooling for chunk derived momentum, per-layer lr mod, decay
MEMORY_MODEL_PER_LAYER_LEARNED_LR = True
NEURAL_MEM_WEIGHT_RESIDUAL = True  # learning to accept contributions from the weights of the previous neural mem layer brings about significant improvements. this was improvised and not in the paper, but inspired by the value residual learning free lunch paper
NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW = True  # will allow the neural memory to select what layers from which to derive queries / keys / values, effectively allowing it to graft itself to the transformer in any way to be beneficial. this is to address an issue from a phd student who noted that the mem network is learning nothing more than wk @ wv. this also generalizes all possible ways to connect the neural memory to a transformer, a sort of NAS
NEURAL_MEM_SPEC_NORM_SURPRISES = True  # applying lessons from Muon optimizer to surprise updates, by spectral norming the surprises

# experiment related

PROJECT_NAME = "titans-mac-transformer"
RUN_NAME = f"mac - {NUM_LONGTERM_MEM} longterm mems, layers {NEURAL_MEM_LAYERS}"
WANDB_ONLINE = False  # turn this on to pipe experiment to cloud

# perf related

USE_ACCELERATED_SCAN = False
USE_FLEX_ATTN = False
USE_FAST_INFERENCE = False

# reproducibility
SEED = 42

# DDP ?
# TODO: DDP setup: called for each subprocess to connect to shared master process
# TODO: Wrap model in DDP so it appears like a local model (when in fact hidden control flow is injected for communication)
# TODO: Save model using model.module (elaborate for safetensors), and only when rank == 0
# TODO: Use custom DistributedSampler in Dataloader (how does cycling work then?)
# TODO: destroy_process_group when training is done
# TODO: Call multiprocessing.spawn with nprocs for automatic rank assignment


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # We want to be able to train our model on an `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`__
    # such as CUDA, MPS, MTIA, or XPU.
    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc)
    # initialize the process group
    init_process_group(backend, rank=rank, world_size=world_size)


# helpers


def decode_token(token):
    return str(chr(max(32, token)))


def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))


def init_model(rank):
    # memory model
    if USE_MEM_ATTENTION_MODEL:
        neural_memory_model = MemoryAttention(dim=64)
    else:
        neural_memory_model = MemoryMLP(dim=64, depth=NEURAL_MEMORY_DEPTH)

    # instantiate memory-as-context transformer

    return DDP(
        MemoryAsContextTransformer(
            num_tokens=256,
            dim=384,
            depth=8,
            segment_len=WINDOW_SIZE,
            num_persist_mem_tokens=NUM_PERSIST_MEM,
            num_longterm_mem_tokens=NUM_LONGTERM_MEM,
            neural_memory_layers=NEURAL_MEM_LAYERS,
            neural_memory_segment_len=NEURAL_MEM_SEGMENT_LEN,
            neural_memory_batch_size=NEURAL_MEM_BATCH_SIZE,
            neural_mem_gate_attn_output=NEURAL_MEM_GATE_ATTN_OUTPUT,
            neural_mem_weight_residual=NEURAL_MEM_WEIGHT_RESIDUAL,
            neural_memory_qkv_receives_diff_views=NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW,
            use_flex_attn=USE_FLEX_ATTN,
            sliding_window_attn=SLIDING_WINDOWS,
            neural_memory_model=neural_memory_model,
            neural_memory_kwargs=dict(
                dim_head=64,
                heads=4,
                attn_pool_chunks=STORE_ATTN_POOL_CHUNKS,
                qk_rmsnorm=NEURAL_MEM_QK_NORM,
                momentum=NEURAL_MEM_MOMENTUM,
                momentum_order=NEURAL_MEM_MOMENTUM_ORDER,
                default_step_transform_max_lr=NEURAL_MEM_MAX_LR,
                use_accelerated_scan=USE_ACCELERATED_SCAN,
                per_parameter_lr_modulation=MEMORY_MODEL_PER_LAYER_LEARNED_LR,
                spectral_norm_surprises=NEURAL_MEM_SPEC_NORM_SURPRISES,
            ),
        ),
        device_ids=[rank],
    )


def precompute_and_shard_indices(data_size: int, seq_len: int, num_indices: int, rank: int, world_size: int) -> torch.Tensor:
    max_start = data_size - seq_len - 1
    indices = torch.randint(0, max_start, (num_indices,))
    # shard precomputed indices
    return indices[rank::world_size]


class TextSamplerDataset(Dataset):
    def __init__(self, data, indices: torch.Tensor, seq_len: int, device: int):
        super().__init__()
        self.data = data
        self.indices = indices
        self.seq_len = seq_len
        self.device = device

    def __getitem__(self, index):
        # rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        start = self.indices[index]
        full_seq = self.data[start : start + self.seq_len + 1].long()
        return full_seq.cuda(self.device)

    def __len__(self):
        return len(self.indices)


# def cycle(loader):
#     while True:
#         for data in loader:
#             yield data


# training
def train(rank: int, world_size: int, update_queue: mp.Queue):
    ddp_setup(rank, world_size)
    # wandb experiment tracker
    # import wandb
    # wandb.init(project=PROJECT_NAME, mode="disabled" if not WANDB_ONLINE else "online")
    # wandb.run.name = RUN_NAME
    # wandb.run.save()

    # prepare enwik8 data
    
    with gzip.open("./data/enwik8.gz") as file:
        data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
        data_train, data_val = np.split(data, [int(90e6)])
        data_train, data_val = map(torch.from_numpy, (data_train, data_val))

    train_indices = precompute_and_shard_indices(len(data_train), SEQ_LEN, NUM_BATCHES * BATCH_SIZE * GRADIENT_ACCUMULATE_EVERY, rank, world_size)
    val_indices = precompute_and_shard_indices(len(data_train), SEQ_LEN, NUM_BATCHES * BATCH_SIZE * GRADIENT_ACCUMULATE_EVERY, rank, world_size)

    train_dataset = TextSamplerDataset(data_train, train_indices, SEQ_LEN, device=rank)
    val_dataset = TextSamplerDataset(data_val, val_indices, SEQ_LEN, device=rank)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
    )
    train_iter, val_iter = iter(train_loader), iter(val_loader)
    model = init_model(rank)
    # optimizer
    optim = AdoptAtan2(model.parameters(), lr=LEARNING_RATE)
    for batch_idx in tqdm(range(NUM_BATCHES // world_size), mininterval=10.0, desc="Pretraining"):
        model.train()
        # each outer batch takes GRADIENT_ACCUMULATE_EVERY steps (inner batch) using the train_iter
        # then each step (next(train_iter)) draw BATCH_SIZE samples from TextSamplerDataset
        # crucially, only do NUM_BATCHES // world_size number of batches
        for _ in range(GRADIENT_ACCUMULATE_EVERY):
            loss = model(next(train_iter), return_loss=True)
            loss.backward()

        print(f"training loss: {loss.item():.4f}") # make into pbar stats
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()
        # wandb.log(dict(loss=loss.item()))

        if rank == 0:
            if (batch_idx + 1) % VALIDATE_EVERY == 0 or batch_idx == 0:
                model.eval()
                with torch.no_grad():
                    loss = model(next(val_iter), return_loss=True)
                    print(f"validation loss: {loss.item():.4f}")

                if (batch_idx + 1) % SAVE_EVERY == 0 or batch_idx == 0 and rank == 0:
                    os.makedirs("checkpoint", exist_ok=True)
                    checkpoint_path = os.path.join(
                        "checkpoint",
                        f"checkpoint_epoch={batch_idx}_valloss={loss.item():.4f}.safetensors",
                    )
                    print(f"\tSaving model to {checkpoint_path}")
                    save_model(model.module, checkpoint_path)
                    # torch.save(model.module.state_dict(), os.path.join("checkpoint", f"checkpoint_epoch={i}_valloss={loss.item:.4f}.pth"))

            if SHOULD_GENERATE and (batch_idx + 1) % GENERATE_EVERY == 0:
                model.eval()
                inp = random.choice(val_dataset)[:PRIME_LENGTH]
                prime = decode_tokens(inp)
                print(f"{prime} \n\n {'*' * 100}")

                sample = model.module.sample(
                    inp[None, ...], GENERATE_LENGTH, use_cache=USE_FAST_INFERENCE
                )
                output_str = decode_tokens(sample[0])
                print(output_str)
        
        update_queue.put((rank, 1))

    destroy_process_group()


def main():
    world_size = torch.cuda.device_count()
    update_queue = mp.Queue()
    with Progress() as progress:
        for i in range(world_size):
            progress.add_task(f"Replica {i}", total = NUM_BATCHES // world_size)

        mp.spawn(train, (world_size, update_queue), nprocs=world_size)

        while not progress.finished:
            if not update_queue.empty():
                msg_task_id, advance_amount = update_queue.get()
                if advance_amount > 0:
                    progress.update(msg_task_id, advance=advance_amount)

            time.sleep(0.001)

if __name__ == "__main__":
    main()
