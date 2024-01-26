from model.model import main
import logging
from transformers import MODEL_WITH_LM_HEAD_MAPPING
from data_loader import preprocess
import pandas as pd
from utils.helper import Args

# Configs
logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

train_df = pd.DataFrame.from_records(preprocess.train_contexted, columns=preprocess.columns)
val_df = pd.DataFrame.from_records(preprocess.validate_contexted, columns=preprocess.columns)

args = Args()

main(df_train=train_df, df_val=val_df, logger=logger, args=args)