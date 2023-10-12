from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from entity import Input, Response
from bartpho_infer import bartpho_summarise
from bartpho_infer_ct2 import bartpho_ct2_summarise

app = FastAPI(title="Length-controllable Summarisation")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/bartpho_summarisation")
async def summarisation(item: Input):
    model = item.model.lower()
    if model not in ["bartpho", "bartpho_ct2"]:
        return Response(status = -1, message = f"model '{model}' not supported")

    content = item.content[0] if item.content else ""
    summary = bartpho_summarise(content, item.no_sentences)

    result = Response(status = 2, message = "success", result=[summary])
    return result


@app.post("/bartpho_ct2_summarisation")
async def summarisation(item: Input):
    content = item.content[0] if item.content else ""
    summary = bartpho_ct2_summarise(content, item.no_sentences)

    result = Response(status = 1, message = "success", result=[summary])
    return result


if __name__ == '__main__':
    uvicorn.run("service_summarisation:app", host="0.0.0.0", port=9874)