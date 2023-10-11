import os
import json

from tqdm import tqdm
import nltk

nltk.download("punkt")

working_dir = "/home2/dungnguyen/length-controllable-summarisation"
with open(os.path.join(working_dir, "data/summary.json")) as f:
    data = json.load(f)

for sample in tqdm(data):
    raw_result_wseg_sentences = nltk.sent_tokenize(sample["result_wseg"])
    len_summary = len(raw_result_wseg_sentences)

    result_wseg_sentences = []
    for idx, sen in enumerate(raw_result_wseg_sentences):
        sen = f"[SN]{idx + 1} " + sen
        result_wseg_sentences.append(sen)
    result_wseg = f"[SN]{len_summary} [SEP]"
    result_wseg = " ".join([result_wseg] + result_wseg_sentences)
    sample["result_wseg"] = result_wseg

    sample["prompt"] = (
        f"Tóm tắt nội dung thông tin dưới đây với {len_summary} câu:"
        + sample["prompt"].split(":", 1)[1]
    )
    sample["prompt_wseg"] = (
        f"Tóm_tắt nội_dung thông_tin dưới đây với {len_summary} câu :"
        + sample["prompt_wseg"].split(" :", 1)[1]
    )

with open(
    os.path.join(working_dir, "data/SentEnum_summary.json"), "w", encoding="utf8"
) as f:
    json.dump(data, f, ensure_ascii=False)
