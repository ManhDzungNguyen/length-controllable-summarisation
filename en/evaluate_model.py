import json
from tqdm import tqdm
import evaluate
import torch
import evaluate
from transformers import AutoTokenizer, T5ForConditionalGeneration
from function.utils import post_process

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
new_special_tokens = ["[SN]", "[SEP]"]  # for example
tokenizer.add_special_tokens({"additional_special_tokens": ["[SN]", "[SEP]"]})
model = T5ForConditionalGeneration.from_pretrained("/home/kuuhaku/work/length-controllable-summarisation/models/sentenum_lengthinstruct/checkpoint-2000")


def load_data(file_path):
    with open(file_path) as f:
        data = json.load(f)

    return data

data = load_data('/home/kuuhaku/work/length-controllable-summarisation/data/SentEnum/no_length_instruction/test.json')  # replace with your file path and file format

rouge = evaluate.load('rouge')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.eval()

decoded_predictions = []
actual_summaries = []


batch_size = 8  # Adjust based on your system's capabilities
for i in tqdm(range(0, len(data), batch_size)):
    batch = data[i:i + batch_size]
    articles = [item['article'] for item in batch]
    actual_summaries.extend([item['highlights'] for item in batch])

    with torch.no_grad():
        inputs = tokenizer(articles, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = inputs.to(device)

        summary_ids = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=250)
        

    decoded_batch_predictions = [post_process(tokenizer.decode(g, skip_special_tokens=False)) for g in summary_ids]
    decoded_predictions.extend(decoded_batch_predictions)

    filename = 'eval_res.txt'

    with open(filename, 'w') as file:
        for item in decoded_predictions:
            file.write(f'{item}\n')

for reference, prediction in zip(actual_summaries, decoded_predictions):
    rouge.add(prediction=prediction, reference=reference)

rouge_results = rouge.compute()


print(rouge_results)