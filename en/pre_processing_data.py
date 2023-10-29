import os
import json
from datasets import load_dataset
from tqdm import tqdm
import nltk

nltk.download("punkt")


working_dir = "/home/kuuhaku/work/length-controllable-summarisation/en/data/CNNDM"
dataset = load_dataset("cnn_dailymail", "3.0.0")
dataset["train"].to_json(os.path.join(working_dir, "train.json"))
dataset["validation"].to_json(os.path.join(working_dir, "validation.json"))
dataset["test"].to_json(os.path.join(working_dir, "test.json"))


def convert_data(dataset, contain_length_instruction=False):
    ls_data = []
    for sample in tqdm(dataset):
        raw_result_wseg_sentences = nltk.sent_tokenize(sample["highlights"])
        len_summary = len(raw_result_wseg_sentences)

        result_wseg_sentences = []
        for idx, sen in enumerate(raw_result_wseg_sentences):
            sen = f"[SN]{idx + 1} " + sen
            result_wseg_sentences.append(sen)
        result_wseg = f"[SN]{len_summary} [SEP]"
        result_wseg = " ".join([result_wseg] + result_wseg_sentences)
        sample["highlights"] = result_wseg

        instruction = (
            f"Summarize the information below in {len_summary} sentences:\n"
            if contain_length_instruction
            else "Summarize the information below:\n"
        )

        sample["article"] = instruction + sample["article"]
        sample["len_summary"] = len_summary
        ls_data.append(sample)

    return ls_data


train_data_no_length_instruction = convert_data(dataset["train"])
validation_data_no_length_instruction = convert_data(dataset["validation"])
test_data_no_length_instruction = convert_data(dataset["test"])

train_data_with_length_instruction = convert_data(
    dataset["train"], contain_length_instruction=True
)
validation_data_with_length_instruction = convert_data(
    dataset["validation"], contain_length_instruction=True
)
test_data_with_length_instruction = convert_data(
    dataset["test"], contain_length_instruction=True
)

working_dir = "/home/kuuhaku/work/length-controllable-summarisation/en/data/CNNDM/SentEnum/no_length_instruction"
with open(os.path.join(working_dir, "train.json"), "w", encoding="utf8") as f:
    json.dump(train_data_no_length_instruction, f, ensure_ascii=False)
with open(os.path.join(working_dir, "validation.json"), "w", encoding="utf8") as f:
    json.dump(validation_data_no_length_instruction, f, ensure_ascii=False)
with open(os.path.join(working_dir, "test.json"), "w", encoding="utf8") as f:
    json.dump(test_data_no_length_instruction, f, ensure_ascii=False)


working_dir = "/home/kuuhaku/work/length-controllable-summarisation/en/data/CNNDM/SentEnum/with_length_instruction"
with open(os.path.join(working_dir, "train.json"), "w", encoding="utf8") as f:
    json.dump(train_data_no_length_instruction, f, ensure_ascii=False)
with open(os.path.join(working_dir, "validation.json"), "w", encoding="utf8") as f:
    json.dump(validation_data_no_length_instruction, f, ensure_ascii=False)
with open(os.path.join(working_dir, "test.json"), "w", encoding="utf8") as f:
    json.dump(test_data_no_length_instruction, f, ensure_ascii=False)
