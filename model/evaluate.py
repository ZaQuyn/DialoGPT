import pandas as pd
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)
from utils.helper import Args
from typing import Dict, List, Tuple
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, SequentialSampler
import torch
import os
from logging import Logger
from tqdm.notebook import tqdm, trange
from utils.helper import load_and_cache_examples


def evaluate(
    args: Args,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    logger: Logger,
    prefix="",
) -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(
        args, tokenizer, df_train, df_val, evaluate=True
    )
    os.makedirs(eval_output_dir, exist_ok=True)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(
            examples, batch_first=True, padding_value=tokenizer.pad_token_id
        )

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=collate,
        drop_last=True,
    )

    # Multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Evaluate
    logger.info(f"***** Running evaluation {prefix} *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size = {args.eval_batch_size}")
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = (batch, batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1
    
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity}

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info(f"***** Eval result {prefix} *****")
        for key in sorted(result.keys()):
            logger.info(f"{key} = {str(result[key])}")
            writer.write("%s = %s\n" % (key, str(result[key])))
    
    return result