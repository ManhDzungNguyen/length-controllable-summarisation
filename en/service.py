from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from entity.types import Input
from function.text_extraction import extract_data
from function.text_summarisation import text_summarisation


app = FastAPI(title="Summarization")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/text_extraction")
async def summarisation(item: Input):
    url = item.url
    content = extract_data(url)
    summary = text_summarisation(
        content, no_tokens=item.no_tokens, no_sens=item.no_sentences
    )
    return {"result": summary}


if __name__ == "__main__":
    uvicorn.run("service:app", host="0.0.0.0", port=6868)
