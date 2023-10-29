import torch
from torch.utils.data import Dataset


class LazyCustomDataset(Dataset):
    def __init__(self, json_file_path, tokenizer):
        self.json_file_path = json_file_path
        self.tokenizer = tokenizer

        with open(self.json_file_path, "r") as file:
            self.data = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source_text = self.data[idx]["article"]
        target_text = self.data[idx]["highlights"]

        encoded_source = self.tokenizer(
            source_text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoded_target = self.tokenizer(
            target_text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoded_source["input_ids"][0]
        attention_mask = encoded_target["attention_mask"][0]
        label = encoded_target["input_ids"][0]

        outp = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
        }
        return outp
