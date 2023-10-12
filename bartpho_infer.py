import os
import json
import warnings
import random

from tqdm import tqdm
import pandas as pd
from py_vncorenlp import VnCoreNLP
from transformers import AutoTokenizer, MBartForConditionalGeneration
import torch

from function.article_handler import clean_article

warnings.filterwarnings("ignore")
random.seed(42)


WORKING_DIR = "/home2/dungnguyen/length-controllable-summarisation"

vi_segmenter = VnCoreNLP(
    save_dir="/home2/dungnguyen/length-controllable-summarisation/models/vncorenlp",
    annotators=["wseg"],
)

model_dir = os.path.join(WORKING_DIR, "models/summary/checkpoint-1100")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = MBartForConditionalGeneration.from_pretrained(model_dir)


def wseg(sent=""):
    sentences = vi_segmenter.word_segment(sent)
    return " ".join(sentences)


def preprocess_context(context):
    context_paras = context.strip().split("\n")
    cleaned_context = ""
    for para in context_paras:
        if len(para.split()) > 9 and "Ảnh:" not in para:
            cleaned_context += para + " "

    return cleaned_context


def generate_prompt(context, no_sen=5):
    instruction = f"Tóm tắt nội dung thông tin dưới đây với {no_sen} câu: \n "
    context = preprocess_context(context)

    prompt = instruction + context
    prompt_wseg = wseg(prompt)

    return prompt_wseg


def bartpho_summarise(context, no_sen=5):
    prompt = generate_prompt(context, no_sen=no_sen)
    input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
    output_ids = model.generate(input_ids, max_length=1024, num_return_sequences=1, num_beams=1)
    result = tokenizer.decode(output_ids[0], skip_special_tokens=False)

    return result, prompt

if __name__ == "__main__":
    with open(
        "/home2/dungnguyen/time-series-event-extraction/archive/articles-sample.json"
    ) as f:
        data = json.load(f)
        data = [
            article
            for article in data
            if article.get("source_type") == 11 and article.get("message") is not None
        ]
        data = random.sample(data, 20)

    summary_data = []
    for article in tqdm(data):
        record = {}

        record["content"] = clean_article(article)

        for no_sen in range(3, 10):
            record[f"sum_{no_sen}"], prompt = bartpho_summarise(record["content"], no_sen)
            # print(f"PROMPT:\n{prompt}")
            # print("----------------")
            # print(f"RESULT:\n{record[f'sum_{no_sen}']}")
            # break
        # break

        summary_data.append(record)

    df = pd.DataFrame(summary_data)
    df.to_excel(
        os.path.join(WORKING_DIR, "data/result/231011_summary_bartpho.xlsx"),
        sheet_name="summary",
        index=False,
    )
