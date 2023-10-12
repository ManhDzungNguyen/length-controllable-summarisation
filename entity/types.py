from pydantic import BaseModel
from typing import List, Optional


class Article(BaseModel):
    id: Optional[str] = ""
    title: Optional[str] = ""
    snippet: Optional[str] = ""
    message: Optional[str] = ""
    source_type: Optional[int] = -1
    url: Optional[str] = ""
    created_time: Optional[int]
    lang: Optional[str] = "vi"


class InputArticles(BaseModel):
    docs: List[Article]


class Input(BaseModel):
    model: str = "bartpho"
    content: List[str]
    no_sentences: int = 5


class Response(BaseModel):
    status: int = 1
    message: str = ""
    result: list = []
