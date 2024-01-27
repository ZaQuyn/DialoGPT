from model.model import main
import logging
from transformers import MODEL_WITH_LM_HEAD_MAPPING
from data_loader import preprocess
import pandas as pd
from utils.argument import Args
import os
import sys


sys.path.append(os.path.abspath("../utils/argument.py"))
# Configs
logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

train_df = pd.DataFrame.from_records(preprocess.train_context, columns=preprocess.columns)
val_df = pd.DataFrame.from_records(preprocess.val_context, columns=preprocess.columns)

args = Args()

main(df_train=train_df, df_val=val_df, logger=logger, args=args)