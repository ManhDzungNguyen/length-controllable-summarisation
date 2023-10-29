import os
import json

from tqdm import tqdm
import evaluate
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
import nltk

nltk.download("punkt")

from function.utils import post_process


calc_score_from_preds_file = True
test_preds_file = "test_predictions.txt"

with open("data/SentEnum/no_length_instruction/test.json") as f:
    test_data = json.load(f)


actual_summaries = [item["highlights"] for item in test_data]
decoded_predictions = []

if calc_score_from_preds_file:
    with open(test_preds_file) as f:
        decoded_predictions = f.readlines()
else:
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    new_special_tokens = ["[SN]", "[SEP]"]
    tokenizer.add_special_tokens({"additional_special_tokens": ["[SN]", "[SEP]"]})

    model = T5ForConditionalGeneration.from_pretrained(
        "/home/kuuhaku/work/length-controllable-summarisation/en/models/sentenum_lengthinstruct/checkpoint-2000"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    batch_size = 8
    for i in tqdm(range(0, len(test_data), batch_size)):
        batch = test_data[i : i + batch_size]
        articles = [item["article"] for item in batch]

        with torch.no_grad():
            inputs = tokenizer(
                articles,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            inputs = inputs.to(device)

            summary_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=250,
            )

        decoded_batch_predictions = [
            post_process(tokenizer.decode(g, skip_special_tokens=False))
            for g in summary_ids
        ]
        decoded_predictions.extend(decoded_batch_predictions)

        with open(test_preds_file, "a", encoding="utf-8") as file:
            for item in decoded_batch_predictions:
                file.write(f"{item}\n")


rouge = evaluate.load("rouge")
true_preds = 0
total_diff = 0

for reference, prediction in zip(actual_summaries, decoded_predictions):
    rouge.add(prediction=prediction, reference=reference)

    no_sen_reference = len(nltk.sent_tokenize(reference))
    accept_length = [no_sen_reference + i for i in range(-1, 2)]
    no_sen_prediction = len(nltk.sent_tokenize(prediction))
    # if no_sen_prediction == no_sen_reference:
    #     true_preds +=1
    if no_sen_prediction in accept_length:
        true_preds += 1
    else:
        total_diff += abs(no_sen_reference - no_sen_prediction)

rouge_results = rouge.compute()
accuracy = true_preds / len(actual_summaries) * 100
diff = total_diff / len(actual_summaries)

print(rouge_results)
print(f"Acc: {accuracy}")
print(f"Dif: {diff}")

# {'rouge1': 0.3823937814307753, 'rouge2': 0.1837885253937466, 'rougeL': 0.28133645001524166, 'rougeLsum': 0.2842237125766617}
# Acc: 59.93037423846823
# Dif: 1.0335073977371627
