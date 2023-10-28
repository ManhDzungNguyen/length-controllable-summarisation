import re
from thefuzz import fuzz


def calculate_similarity(str1: str, str2: str):
    return fuzz.token_sort_ratio(str1, str2) / 100.0


def transform_serial_numbers(text):
    pattern = r"\[SN\]\s*\d+"

    matches = re.finditer(pattern, text)

    transformed_matches = []
    for match in matches:
        match_text = match.group(0)
        transformed_text = re.sub(r"\]\s*", "]", match_text)

        start, end = match.span()
        transformed_matches.append((start, end, transformed_text))

    new_text = text
    for start, end, transformed_text in reversed(transformed_matches):
        new_text = new_text[:start] + transformed_text + new_text[end:]

    return new_text


def post_process(raw_summary, remove_duplicated_sentences=False):
    if raw_summary.startswith("<pad>"):
        raw_summary = raw_summary[5:]

    if raw_summary.endswith("</s>"):
        raw_summary = raw_summary[:-4]

    # try:
    prefix, root = raw_summary.split("[SEP]", 1)

    # handle prefix
    pattern = r"\[SN\]\s*(\d+)"
    matches = re.findall(pattern, prefix)
    if len(matches) != 1:
        return raw_summary
    no_required_sentences = int(matches[0])

    # handle root:
    sentences = root.split("[SN]")
    sentences = sentences[: no_required_sentences + 1]
    if len(sentences) < 2:
        return raw_summary

    if remove_duplicated_sentences:
        new_sentences = [sentences[0]]
        for sen in sentences[1:]:
            if calculate_similarity(new_sentences[-1], sen) < 0.75:
                new_sentences.append(sen)
        sentences = new_sentences

    root = "[SN]".join(sentences)

    summary = "[SEP]".join([prefix, root])
    summary = transform_serial_numbers(summary)
    return summary
