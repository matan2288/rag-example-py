"""Ranking self-check: no API. A chunk identical to the query must rank first."""

from knowledge import PieceOfText, how_similar


def closest_text(query_numbers, pieces):
    return max(pieces, key=lambda p: how_similar(query_numbers, p.numbers)).text


def main():
    query = [1.0, 0.0, 0.0]
    pieces = [
        PieceOfText("about cars", "a.md", [0.0, 1.0, 0.0]),
        PieceOfText("about Alex", "b.md", [1.0, 0.0, 0.0]),
        PieceOfText("about homes", "c.md", [0.0, 0.0, 1.0]),
    ]
    assert closest_text(query, pieces) == "about Alex"
    print("ok")


if __name__ == "__main__":
    main()
