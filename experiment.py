"""Fixed CPU experiment: 32 input bytes, one unseen next-byte target."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from prepare import DATA_SHA256, read_verified_data
import train

SEED = 42
CPU_THREADS = 2
CONTEXT_LENGTH = 32
TRAIN_SECONDS = 10.0
VOCAB_SIZE = 256
VALIDATION_EXAMPLES = 4096
ROOT = Path(__file__).parent


def logits_for(model, inputs):
    logits = model(inputs)
    if logits.shape != (len(inputs), VOCAB_SIZE):
        raise ValueError(f"Expected [batch, 256] logits; received {tuple(logits.shape)}")
    return logits


@torch.no_grad()
def evaluate(model, validation_data):
    # Spread fixed windows across the entire validation split.
    starts = torch.linspace(
        0, len(validation_data) - CONTEXT_LENGTH - 1, VALIDATION_EXAMPLES
    ).long()
    model.eval()
    total_nll = 0.0
    for batch_starts in starts.split(128):
        positions = batch_starts[:, None] + torch.arange(CONTEXT_LENGTH)
        inputs = validation_data[positions]
        targets = validation_data[batch_starts + CONTEXT_LENGTH]
        total_nll += F.cross_entropy(
            logits_for(model, inputs), targets, reduction="sum"
        ).item()
    return total_nll / len(starts) / math.log(2)


def run_experiment():
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(1)
    random.seed(SEED)
    torch.manual_seed(SEED)
    candidate_sha = hashlib.sha256((ROOT / "train.py").read_bytes()).hexdigest()
    corpus = torch.tensor(list(read_verified_data()), dtype=torch.long)
    split = int(0.9 * len(corpus))
    training_data, validation_data = corpus[:split], corpus[split:]
    model = train.build_model(VOCAB_SIZE, CONTEXT_LENGTH).cpu()
    optimizer = train.build_optimizer(model)
    batch_size = train.BATCH_SIZE
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("BATCH_SIZE must be a positive integer.")
    generator = torch.Generator().manual_seed(SEED + 1)
    offsets = torch.arange(CONTEXT_LENGTH)
    model.train()
    steps = 0
    start = time.perf_counter()
    while time.perf_counter() - start < TRAIN_SECONDS:
        starts = torch.randint(
            0, len(training_data) - CONTEXT_LENGTH, (batch_size,), generator=generator
        )
        inputs = training_data[starts[:, None] + offsets]
        targets = training_data[starts + CONTEXT_LENGTH]
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(logits_for(model, inputs), targets)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss.")
        loss.backward()
        optimizer.step()
        steps += 1
    training_seconds = time.perf_counter() - start
    val_bpb = evaluate(model, validation_data)
    if not math.isfinite(val_bpb):
        raise ValueError("Non-finite validation loss.")
    return {
        "val_bpb": val_bpb,
        "training_seconds": training_seconds,
        "steps": steps,
        "num_params": sum(p.numel() for p in model.parameters()),
        "candidate_sha256": candidate_sha,
        "data_sha256": DATA_SHA256,
        "seed": SEED,
        "context_length": CONTEXT_LENGTH,
        "train_budget_seconds": TRAIN_SECONDS,
        "batch_size": batch_size,
        "validation_examples": VALIDATION_EXAMPLES,
        "device": "cpu",
        "cpu_threads": CPU_THREADS,
        "torch_version": torch.__version__,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New result JSON path")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Result already exists: {args.output}. Choose a new output path.")
    result = run_experiment()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, allow_nan=False)
    args.output.write_text(rendered + "\n")
    print(rendered)
