import pickle
from dataclasses import dataclass
from pathlib import Path

KB_FOLDER = Path(__file__).parent / "knowledge-base"
STORE_FILE = Path(__file__).parent / "data" / "store.pkl"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 200
HOW_MANY = 10


@dataclass
class PieceOfText:
    text: str
    source: str
    numbers: list[float]


def markdown_files() -> list[Path]:
    return sorted(KB_FOLDER.rglob("*.md"))


def cut_into_chunks(text: str) -> list[str]:
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    start = 0
    while start < len(text):
        chunks.append(text[start : start + CHUNK_SIZE])
        start += step
    return chunks


def how_similar(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb + 1e-9)


def _build_store() -> list[PieceOfText]:
    from openai_client import turn_into_numbers_many

    pieces: list[PieceOfText] = []
    for path in markdown_files():
        text = path.read_text(encoding="utf-8")
        for chunk in cut_into_chunks(text):
            pieces.append(PieceOfText(text=chunk, source=str(path), numbers=[]))

    print(f"Embedding {len(pieces)} chunks (first run only)...")
    vectors = turn_into_numbers_many([p.text for p in pieces])
    for piece, vector in zip(pieces, vectors):
        piece.numbers = vector

    STORE_FILE.parent.mkdir(exist_ok=True)
    STORE_FILE.write_bytes(pickle.dumps(pieces))
    return pieces


def load_knowledge() -> list[PieceOfText]:
    if STORE_FILE.exists():
        return pickle.loads(STORE_FILE.read_bytes())
    return _build_store()


def find_closest(question: str, pieces: list[PieceOfText], how_many: int = HOW_MANY) -> list[PieceOfText]:
    from openai_client import turn_into_numbers

    question_numbers = turn_into_numbers(question)
    ranked = sorted(
        pieces,
        key=lambda piece: how_similar(question_numbers, piece.numbers),
        reverse=True,
    )
    return ranked[:how_many]
