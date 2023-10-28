from transformers import AutoTokenizer, T5ForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
new_special_tokens = ["[SN]", "[SEP]"]  # for example
tokenizer.add_special_tokens({"additional_special_tokens": ["[SN]", "[SEP]"]})
model = T5ForConditionalGeneration.from_pretrained("./models/sentenum_lengthinstruct/checkpoint-1200")

input_ids = tokenizer("Summarize the information below in 1 sentences:\nZully Broussard's donation was a big deal, but she's a big donor.",
                      return_tensors="pt",
                      truncation=True).input_ids
outputs = model.generate(input_ids, max_new_tokens=150)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))