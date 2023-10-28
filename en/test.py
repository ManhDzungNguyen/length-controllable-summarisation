# from transformers import AutoTokenizer, T5ForConditionalGeneration

# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
# new_special_tokens = ["[SN]", "[SEP]"]  # for example
# tokenizer.add_special_tokens({"additional_special_tokens": ["[SN]", "[SEP]"]})
# model = T5ForConditionalGeneration.from_pretrained("./models/sentenum_lengthinstruct/checkpoint-1200")

# input_ids = tokenizer("Summarize the information below in 1 sentences:\nZully Broussard's donation was a big deal, but she's a big donor.",
#                       return_tensors="pt",
#                       truncation=True).input_ids
# outputs = model.generate(input_ids, max_new_tokens=150)
# print(tokenizer.decode(outputs[0], skip_special_tokens=False))

import re

def transform_serial_numbers(text):
    # Define the regular expression pattern to match '[SN] ' followed by numbers (with any amount of space)
    pattern = r'\[SN\]\s*\d+'

    # Find all matches of the pattern in the text
    matches = re.finditer(pattern, text)

    # For each match, create a new string where the spaces are removed
    transformed_matches = []
    for match in matches:
        # Get the matched string
        match_text = match.group(0)

        # Remove spaces: specifically, remove spaces between ']' and digits
        transformed_text = re.sub(r'\]\s*', ']', match_text)

        # Store the transformed match along with the start and end indices
        start, end = match.span()
        transformed_matches.append((start, end, transformed_text))

    # Now, construct the new text by replacing the original matches with the transformed ones
    # We replace from end to start to avoid messing up the indices
    new_text = text
    for start, end, transformed_text in reversed(transformed_matches):
        new_text = new_text[:start] + transformed_text + new_text[end:]

    return new_text

# Test the function with some text
input_text = "Text before [SN]   12345 and some text in between [SN]  67890 and text after."
transformed_text = transform_serial_numbers(input_text)

print(transformed_text)