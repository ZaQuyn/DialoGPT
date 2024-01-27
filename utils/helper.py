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
from argument import Args


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


def sorted_checkpoint(
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


def rotate_checkpoints(
    args: Args, checkpoint_prefix="checkpoint", use_mtime=False, logger: Logger = None
) -> None:
    if not args.save_total_limit:
        return None
    if args.save_total_limit <= 0:
        return None

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = sorted_checkpoint(args, checkpoint_prefix, use_mtime)
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
