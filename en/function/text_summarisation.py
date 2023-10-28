from transformers import AutoTokenizer, T5ForConditionalGeneration
from .utils import post_process

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
new_special_tokens = ["[SN]", "[SEP]"]  # for example
tokenizer.add_special_tokens({"additional_special_tokens": ["[SN]", "[SEP]"]})
model = T5ForConditionalGeneration.from_pretrained(
    "/home/kuuhaku/work/length-controllable-summarisation/models/sentenum_lengthinstruct/checkpoint-2000"
)


def text_summarisation(content: str, no_tokens=150, no_sens=None) -> str:
    instruct = (
        f"Summarize the information below in {no_sens} sentences:\n"
        if no_sens
        else "Summarize the information below:\n"
    )
    input_text = "Summarize the information below:\n" + content
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True).input_ids
    outputs = model.generate(input_ids, max_new_tokens=no_tokens)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=False)

    return post_process(summary)
