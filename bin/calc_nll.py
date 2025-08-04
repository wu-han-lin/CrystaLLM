"""
Adapted from:
https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/train.py
"""
import os
from dataclasses import dataclass
from typing import Union
import math
import time

from tqdm import trange

from crystallm import parse_config
from omegaconf import OmegaConf
import numpy as np
import torch
import pickle
from contextlib import nullcontext

from crystallm import (
    GPT,
    GPTConfig,
)


@dataclass
class TrainDefaults:
    out_dir: str = "out"  # the path to the folder where the model checkpoints will be stored
    eval_interval: int = 250  # how often to evaluate against the validation set
    log_interval: int = 1  # how often to print to
    eval_iters_train: int = 200
    eval_iters_val: int = 200
    eval_only: bool = False  # if True, script exits right after the first eval
    always_save_checkpoint: bool = False  # if True, always save a checkpoint after each eval
    init_from: str = "scratch"  # 'scratch' or 'resume'

    # data
    dataset: str = ""  # the path to the folder containing the .bin files with encoded tokens
    gradient_accumulation_steps: int = 40  # used to simulate larger batch sizes
    batch_size: int = 64  # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size: int = 2048  # context of up to `block_size` previous characters

    # model
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
    bias: bool = False  # do we use bias inside LayerNorm and Linear layers?

    # AdamW optimizer
    learning_rate: float = 6e-4  # max learning rate
    max_iters: int = 600000  # total number of training iterations
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95  # make a bit bigger because number of tokens per iter is small
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0

    # learning rate decay settings
    decay_lr: bool = True  # whether to decay the learning rate
    warmup_iters: int = 2000  # how many steps to warm up for; not super necessary potentially
    lr_decay_iters: int = 600000  # should be ~= max_iters per Chinchilla
    min_lr: float = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    # system
    device: str = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = "bfloat16"  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile: bool = True  # use PyTorch 2.0 to compile the model to be faster
    underrep_p: float = 0.0
    validate: bool = False  # whether to evaluate the model using the validation set


def read_start_indices(
    max_start_index: int,
    data_dir: str,
    starts_fname: str,
    on_condition: bool = True,
    required: bool = False,
) -> Union[torch.Tensor, None]:
    start_indices = None
    starts_path = os.path.join(data_dir, starts_fname)
    if on_condition:
        if os.path.exists(starts_path):
            print(f"Reading start indices from {starts_path}...")
            with open(starts_path, "rb") as f:
                start_indices = torch.tensor(pickle.load(f))  # should be sorted
            # remove indices that would result in out-of-bounds sequences
            start_indices = start_indices[start_indices <= max_start_index]
        elif required:
            raise Exception(f"Expected to find a file in dataset dir named '{starts_fname}'")
    return start_indices


if __name__ == "__main__":
    C = parse_config(TrainDefaults)

    # print("Using configuration:")
    # print(OmegaConf.to_yaml(C))

    print(f"Creating {C.out_dir}...")
    os.makedirs(C.out_dir, exist_ok=True)

    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = "cuda" if "cuda" in C.device else "cpu"  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[C.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    if not C.dataset:
        raise Exception("The 'dataset' option is required and cannot be empty")

    train_data = np.memmap(os.path.join(C.dataset, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(C.dataset, "val.bin"), dtype=np.uint16, mode="r") if C.validate else None

    cif_start_indices = read_start_indices(
        max_start_index=len(train_data) - C.block_size,
        data_dir=C.dataset,
        starts_fname="starts.pkl",
    )

    cif_start_indices_val = read_start_indices(
        max_start_index=(len(val_data) - C.block_size) if C.validate else -1,
        data_dir=C.dataset,
        starts_fname="starts_val.pkl",
        on_condition=C.validate,
    )

    cif_start_indices_underrep = read_start_indices(
        max_start_index=len(train_data) - C.block_size,
        data_dir=C.dataset,
        starts_fname="starts_underrep.pkl",
        on_condition=C.underrep_p > 0,
        required=True,
    )

    def get_batch(split):
        data = train_data if split == "train" else val_data

        ix = torch.randint(len(data) - C.block_size, (C.batch_size,))
        if split == "train":
            if C.underrep_p is not None and np.random.random() < C.underrep_p:
                ix = cif_start_indices_underrep[torch.randperm(len(cif_start_indices_underrep))[:C.batch_size]]
            elif cif_start_indices is not None:
                ix = cif_start_indices[torch.randperm(len(cif_start_indices))[:C.batch_size]]
        elif cif_start_indices_val is not None:
            ix = cif_start_indices_val[torch.randperm(len(cif_start_indices_val))[:C.batch_size]]

        x = torch.stack([torch.from_numpy((data[i:i + C.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + C.block_size]).astype(np.int64)) for i in ix])

        if device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(C.device, non_blocking=True), y.pin_memory().to(C.device, non_blocking=True)
        else:
            x, y = x.to(C.device), y.to(C.device)
        return x, y

    iter_num = 0
    best_val_loss = 1e9

    meta_path = os.path.join(C.dataset, "meta.pkl")
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta_vocab_size = meta["vocab_size"]
        print(f"Found vocab_size = {meta_vocab_size} (inside {meta_path})")

    model_args = dict(n_layer=C.n_layer, n_head=C.n_head, n_embd=C.n_embd, block_size=C.block_size,
                      bias=C.bias, vocab_size=None, dropout=C.dropout)
    if C.init_from == "scratch":
        print("Initializing a new model from scratch...")
        if meta_vocab_size is None:
            print("Defaulting to vocab_size of 371...")
        model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 371
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif C.init_from == "resume":
        print(f"Resuming training from {C.out_dir}...")
        ckpt_path = os.path.join(C.out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=C.device)
        checkpoint_model_args = checkpoint["model_args"]
        # force these config attributes to be equal otherwise we can't even resume training;
        #  the rest of the attributes (e.g. dropout) can stay as desired
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = checkpoint_model_args[k]
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
        # print("best_val_loss", best_val_loss)
    # crop down the model block size if desired, using model surgery
    if C.block_size < model.config.block_size:
        model.crop_block_size(C.block_size)
        model_args["block_size"] = C.block_size  # so that the checkpoint will have the right value
    model.to(C.device)

    # initialize a GradScaler; if enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(C.dtype == "float16"))
    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_nll():
        out = {}
        model.eval()
        eval_iters = 10000 // C.batch_size
        for split, eval_iters in [("val", eval_iters)]:
            losses = torch.zeros(eval_iters)
            for k in trange(eval_iters, desc=f"Evaluating NLL"):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        # model.train()
        return out

    losses = estimate_nll()
    print('*'*100)
    print(f"Negative Log Likelihood: {losses['val']:.4f}")
    print('*'*100)
    
