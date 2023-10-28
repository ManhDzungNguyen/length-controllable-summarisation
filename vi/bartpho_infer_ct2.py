import os
import json
import time
import random

from tqdm import tqdm
import pandas as pd
from py_vncorenlp import VnCoreNLP
from transformers import AutoTokenizer
import ctranslate2

from function.article_handler import clean_article

random.seed(42)


WORKING_DIR = "/home2/dungnguyen/length-controllable-summarisation"

vi_segmenter = VnCoreNLP(
    save_dir=os.path.join(WORKING_DIR, "models/vncorenlp"), annotators=["wseg"]
)

tokenizer_dir = os.path.join(WORKING_DIR, "models/summary/bartpho/tokenizer")
model_dir = os.path.join(WORKING_DIR, "models/quantized/bartpho/ct2_model_checkpoint-1100")

tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
model = ctranslate2.Translator(model_dir, compute_type="int8", device="cuda")


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


def bartpho_ct2_summarise(context, no_sen=5, beam_size=5, num_outputs=1):
    beam_size = max(beam_size, num_outputs)
    prompt = generate_prompt(context, no_sen=no_sen)
    input_tokens = tokenizer.convert_ids_to_tokens(
        tokenizer.encode(prompt, max_length=1024, padding=True, truncation=True)
    )

    outputs = model.translate_batch(
        source=[input_tokens],
        beam_size=beam_size,
        num_hypotheses=num_outputs,
        no_repeat_ngram_size=3,
        # max_decoding_length=5
    )
    result = []

    for i in range(num_outputs):
        target = outputs[0].hypotheses[i]
        x = tokenizer.decode(
            tokenizer.convert_tokens_to_ids(target), skip_special_tokens=False
        )
        result.append(x)

    return result


if __name__ == "__main__":
    # start_time = time.time()
    # context = ""
    # print(generate_ct2(context, no_sen=5))
    # print(f"\nRuntime: {time.time() - start_time}")

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
            record[f"sum_{no_sen}"] = bartpho_ct2_summarise(
                context=record["content"], no_sen=no_sen
            )

        summary_data.append(record)

    df = pd.DataFrame(summary_data)
    df.to_excel(
        os.path.join(WORKING_DIR, "data/result/231011_summary_ct2_bartpho.xlsx"),
        sheet_name="summary",
        index=False,
    )
