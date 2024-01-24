import pandas as pd
from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
)
from utils.helper import Args
import torch
import os
import logging
from logging import Logger
from utils.helper import sorted_checkpoint, set_seed, load_and_cache_examples
from train import train
import glob
from evaluate import evaluate


def main(df_train: pd.DataFrame, df_val: pd.DataFrame, logger: Logger):
    args = Args()

    if args.should_continue:
        sorted_checkpoints = sorted_checkpoint(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError(
                "Used --should_continue but no checkpoint was found in --output_dir."
            )
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
        and not args.should_continue
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        f"Process rank: {args.local_rank}, device: {device}, n_gpu: {args.n_gpu}, distributed training: {bool(args.local_rank != -1)}, 16-bits training: {args.fp16}"
    )

    # Set seed
    set_seed(args)

    config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    model = AutoModelWithLMHead.from_pretrained(
        args.model_name_or_path,
        from_tf=False,
        config=config,
        cache_dir=args.cache_dir
    )
    model.to(args.device)

    logger.info(f"Training/evaluation parameters {args}")

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, df_train, df_val, evaluate=False)

        global_step, train_loss = train(args, train_dataset, model, tokenizer)
        logger.info(f"  global_step = {global_step}, average loss = {train_loss}")

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train:
        # Create output directory if needed
        os.makedirs(args.output_dir, exist_ok=True)

        logger.info(f"Saving model checkpoint to {args.output_dir}")
        # Save a trained model, configuration and tokenizer using `save_pretrained()`
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        ) # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = AutoModelWithLMHead.from_pretrained(args.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN) # Reduce logging
        logger.info(f"Evaluate the following checkpoints: {checkpoints}")
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = AutoModelWithLMHead.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, df_train, df_val, prefix=prefix)
            result = dict((k + f"_{global_step}", v) for k, v in result.items())
            result.update(result)
        
    return result