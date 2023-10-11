import os

import pandas as pd
from sklearn.model_selection import train_test_split
# import wandb

os.environ["WANDB_PROJECT"]="length-controllable-summarisation"
os.environ["WANDB_LOG_MODEL"]="false"
os.environ["WANDB_WATCH"]="false"

from transformers import (
    AutoTokenizer,
    MBartForConditionalGeneration,
    AutoConfig,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import Dataset
import torch

WORKING_DIR = "/home2/dungnguyen/length-controllable-summarisation"


class CustomDataset(Dataset):
    def __init__(self, source, target):
        self.source = source
        self.target = target

    def __len__(self):
        return len(self.source["input_ids"])

    def __getitem__(self, idx):
        #         s = self.source[idx]
        #         t = self.target[idx]
        input_ids = self.source["input_ids"][idx]
        attention_mask = self.source["attention_mask"][idx]
        label = self.target["input_ids"][idx]
        outp = {}
        outp["input_ids"] = input_ids
        outp["attention_mask"] = attention_mask
        outp["labels"] = label
        return outp


def get_model(pretrained_model="vinai/bartpho-word"):
    config = AutoConfig.from_pretrained(pretrained_model, dropout=0.3, use_cache=False)
    print(config)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model = MBartForConditionalGeneration.from_pretrained(
        pretrained_model, config=config
    )

    tokenizer.add_special_tokens({"additional_special_tokens": ["[SN]", "[SEP]"]})
    model.resize_token_embeddings(len(tokenizer))

    assert (
        len(tokenizer) == model.get_input_embeddings().weight.shape[0]
    ), "Tokenizer vocabulary size does not match model input embedding size"
    return model, tokenizer


def get_data(input_file):
    df = pd.read_json(input_file)
    print(len(df))
    source = []
    target = []
    for i in df.index:
        source.append(df.loc[i].prompt_wseg)
        target.append(df.loc[i].result_wseg)

    train_target, eval_target, train_source, eval_source = train_test_split(
        target, source, test_size=0.02, random_state=42
    )
    return train_target, eval_target, train_source, eval_source


def process_data(input_file, tokenizer):
    def tokenize_data(data):
        return tokenizer(
            data,
            max_length=1024,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    train_target, eval_target, train_source, eval_source = get_data(input_file)
    train_source_tokenized = tokenize_data(train_source)
    train_target_tokenized = tokenize_data(train_target)
    eval_source_tokenized = tokenize_data(eval_source)
    eval_target_tokenized = tokenize_data(eval_target)

    train_dataset = CustomDataset(train_source_tokenized, train_target_tokenized)
    eval_dataset = CustomDataset(eval_source_tokenized, eval_target_tokenized)

    return train_dataset, eval_dataset


def train(pretrained_model, input_file, saved_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_model(pretrained_model)

    tokenizer.save_pretrained(os.path.join(WORKING_DIR, "models/summary/tokenizer"))

    train_dataset, eval_dataset = process_data(input_file, tokenizer)
    model.to(device)

    training_args = TrainingArguments(
        output_dir=saved_dir,
        report_to="wandb",
        evaluation_strategy="steps",
        learning_rate=2e-5,
        num_train_epochs=8,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        weight_decay=0.01,
        save_total_limit=5,
        load_best_model_at_end=True,
        save_strategy="steps",
        eval_steps=25,
        save_steps=50,
        gradient_accumulation_steps=64,
        warmup_steps=20,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()


if __name__ == "__main__":
    train(
        pretrained_model="vinai/bartpho-word",
        input_file=os.path.join(WORKING_DIR, "data/SentEnum_summary.json"),
        saved_dir=os.path.join(WORKING_DIR, "models/summary"),
    )
