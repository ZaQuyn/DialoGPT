from transformers import AutoModelWithLMHead, AutoTokenizer
import torch
import json
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-small")

train_data = json.load(open("../data/train_data.json"))
val_data = json.load(open("../data/validate_data.json"))

train_context = [[train_data[i][1], train_data[i][0]] for i in range(len(train_data))]
val_context = [[val_data[i][1], val_data[i][0]] for i in range(len(val_data))]

columns = ["response", "context"]
columns = columns + ["context/" + str(i) for i in range(0)]