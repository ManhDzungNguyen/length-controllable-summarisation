from .utils import normalize_text

def clean_article(article):
    title = article.get("title", "")
    title = normalize_text(title)

    snippet = article.get("snippet", "")
    snippet = normalize_text(snippet)

    message = article.get("message", "")
    message = normalize_text(message)

    summary_content = " ".join([title, snippet, message])

    return summary_content