import os
import warnings

import torch
from pytorch_lightning.loggers import WandbLogger


# prep script to run before initializing COMET modules
def prep() -> None:
    # disable warnings
    warnings.simplefilter("ignore")

    # set torch flag float32_matmul_precision
    # (https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html)
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        if device_name in ["NVIDIA GeForce RTX 3090", "NVIDIA RTX A6000"]:
            # print(f"GPU is {device_name}, setting float32 matmul precision to `medium`.")
            torch.set_float32_matmul_precision("medium")

# default function for `json.dump` in case of non-serializable data types
def default_jsonify(elem):
    if type(elem) == WandbLogger:
        return str(elem) # TODO
    else:
        return elem.__dict__

# returns WandbLogger with my standard parameters
def get_wandb_logger() -> WandbLogger:
    if not os.environ.get("WANDB_MODE") == "disabled":
        return WandbLogger(
            project=os.getenv("PROJECT"),
            name=os.getenv("RUN"),
            log_model=False,
        )
    else:
        print("ATTENTION: wandb is disabled through WANDB_MODE env! get_wandb_logger() returning null")
        return None