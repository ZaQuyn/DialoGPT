import torch
from transformers import PreTrainedTokenizer
import pandas as pd
from data_loader.data_loader import ConversationDataset
import random
import numpy as np
from typing import List
import glob
import os
import re
from logging import Logger
import shutil


class Args:
    def __init__(self):
        self.output_dir = "output_small_save"
        self.model_type = "gpt2"
        self.model_name_or_path = "microsoft/DialoGPT-small"
        self.config_name = "microsoft/DialoGPT-small"
        self.tokenizer_name = "microsoft/DialoGPT-small"
        self.cache_dir = "cached"
        self.block_size = 512
        self.do_train = True
        self.do_eval = True
        self.evaluate_during_training = False
        self.per_gpu_train_batch_size = 4
        self.per_gpt_eval_batch_size = 4
        self.gradient_accumulation_steps = 1
        self.learning_rate = 5e-5
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.num_train_epochs = 3
        self.max_steps = -1
        self.warmup_steps = 0
        self.logging_steps = 1000
        self.save_steps = 3500
        self.save_total_limit = None
        self.eval_all_checkpoints = False
        self.no_cuda = False
        self.overwrite_output_dir = True
        self.overwrite_cache = True
        self.should_continue = False
        self.seed = 42
        self.local_rank = -1
        self.fp16 = False
        self.fp16_opt_level = "O1"


# Cacheing and storing of data/checkpoints
def load_and_cache_examples(
    args: Args,
    tokenizer: PreTrainedTokenizer,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    evaluate=False,
):
    return ConversationDataset(tokenizer, args, df_val if evaluate else df_train)


def set_seed(args: Args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _sorted_checkpoint(
    args: Args, checkpoint_prefix="checkpoint", use_mtime=False
) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(
        os.path.join(args.output_dir, f"{checkpoint_prefix}-*")
    )

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append(
                    (int(regex_match.groups()[0]), path)
                )

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(
    args: Args, checkpoint_prefix="checkpoint", use_mtime=False, logger: Logger = None
) -> None:
    if not args.save_total_limit:
        return None
    if args.save_total_limit <= 0:
        return None

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoint(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return None

    number_of_checkpoints_to_delete = max(
        0, len(checkpoints_sorted) - args.save_total_limit
    )
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(
            f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit"
        )
        shutil.rmtree(checkpoint)