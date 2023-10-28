from pydantic import BaseModel
from typing import List, Optional


class Input(BaseModel):
    url: str = "https://paperswithcode.com/task/text-summarization"
    no_sentences: int = 5
    no_tokens: int = 150

