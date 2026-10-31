import os
from dotenv import load_dotenv

load_dotenv(override=True)

CHAT_MODEL = "gpt-4.1-nano"
EMBED_MODEL = "text-embedding-3-large"


def _client():
    from openai import OpenAI

    return OpenAI()


def require_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in a .env file")


def turn_into_numbers(text: str) -> list[float]:
    response = _client().embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding


def turn_into_numbers_many(texts: list[str]) -> list[list[float]]:
    vectors = []
    for start in range(0, len(texts), 100):
        batch = texts[start : start + 100]
        response = _client().embeddings.create(model=EMBED_MODEL, input=batch)
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors.extend(item.embedding for item in ordered)
    return vectors


def answer_from_context(question: str, history: list[dict], context: str) -> str:
    system = f"""You are a knowledgeable, friendly assistant representing Insurellm.
If relevant, use the given context to answer. If you don't know, say so.
Context:
{context}
"""
    messages = [{"role": "system", "content": system}, *history, {"role": "user", "content": question}]
    response = _client().chat.completions.create(
        model=CHAT_MODEL, messages=messages, temperature=0
    )
    return response.choices[0].message.content or ""
