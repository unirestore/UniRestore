# command: fit, validate, test, predict
# main.py
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.cli import LightningCLI

# handle the version conflict
import collections
collections.Iterable = collections.abc.Iterable

import numpy as np
np.float_ = np.float64

import os, sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def cli_main():
    cli = LightningCLI(auto_configure_optimizers=False)

    # maunally call fit
    # cli.trainer.validate(cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("medium")

    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block
