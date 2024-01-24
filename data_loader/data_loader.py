import pandas as pd
from transformers import PreTrainedTokenizer
from typing import List

import torch
from torch.utils.data import Dataset
import os
import logging
import pickle
import sys

from utils.helper import Args

# Config logger
logger = logging.getLogger(name=__name__)


class ConversationDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        args: Args,
        df: pd.DataFrame,
        block_size: int = 512,
    ) -> None:
        super().__init__()

        block_size = block_size - (
            tokenizer.model_max_length - tokenizer.max_len_single_sentence
        )

        directory = args.cache_dir
        cached_features_file = os.path.join(
            directory, args.model_type + "_cached_lm_" + str(block_size)
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info(f"Loading features from cached file {cached_features_file}")
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info(f"Creating features from dataset file at {directory}")

            self.examples = []
            for _, row in df.iterrows():
                flatten = lambda l: [item for sublist in l for item in sublist]
                convers = list(
                    reversed(
                        [tokenizer.encode(x) + [tokenizer.eos_token_id] for x in row]
                    )
                )
                convers = flatten(convers)
                self.examples.append(convers)

            logger.info(f"Saving features into cached file {cached_features_file}")
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return torch.tensor(self.examples[index], dtype=torch.long)
