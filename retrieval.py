"""Small TF-IDF retrieval demo with no third-party dependencies."""

import argparse
import json
import math
from pathlib import Path
import re
from collections import Counter


DEFAULT_DOCUMENTS = Path(__file__).parent / "documents.json"


def tokenize(text):
    """Extract case-insensitive words; punctuation is ignored."""
    return re.findall(r"\b\w+\b", text.lower())


def cosine_similarity(left, right):
    """Compare sparse vectors, returning zero for an empty vector."""
    dot = sum(value * right.get(term, 0.0) for term, value in left.items())
    norm_left = math.sqrt(sum(value * value for value in left.values()))
    norm_right = math.sqrt(sum(value * value for value in right.values()))
    return dot / (norm_left * norm_right) if norm_left and norm_right else 0.0


class Retriever:
    def __init__(self, documents):
        self.documents = list(documents)
        counts = [Counter(tokenize(doc["text"])) for doc in self.documents]
        frequency = Counter(term for count in counts for term in count)
        # Smoothed IDF keeps terms appearing in every document meaningful.
        self.idf = {
            term: math.log((1 + len(counts)) / (1 + number)) + 1
            for term, number in frequency.items()
        }
        self.vectors = [self._vector(count) for count in counts]

    def _vector(self, counts):
        # Raw term frequency; cosine normalization removes vector length effects.
        return {term: count * self.idf[term] for term, count in counts.items()
                if term in self.idf}

    def search(self, query, top_k=3):
        """Return (document, score) pairs, excluding zero-score matches.

        Ties preserve collection order. Unknown query terms are ignored.
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_vector = self._vector(Counter(tokenize(query)))
        matches = [(doc, cosine_similarity(query_vector, vector))
                   for doc, vector in zip(self.documents, self.vectors)]
        return sorted((match for match in matches if match[1] > 0),
                      key=lambda match: match[1], reverse=True)[:top_k]


def load_documents(path):
    documents = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(documents, list) or any(
        not isinstance(doc, dict)
        or any(not isinstance(doc.get(key), str) for key in ("id", "title", "text"))
        for doc in documents
    ):
        raise ValueError("documents must be a JSON list with string id, title, and text fields")
    return documents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="query text (quote queries containing spaces)")
    parser.add_argument("--top-k", type=int, default=3, help="maximum results (default: 3)")
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS,
                        help="path to a JSON document collection")
    args = parser.parse_args()
    try:
        matches = Retriever(load_documents(args.documents)).search(args.query, args.top_k)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not matches:
        print("No matching documents.")
    for position, (doc, score) in enumerate(matches, start=1):
        print(f"{position}. {doc['title']} [{doc['id']}] (score: {score:.4f})")
        print(doc["text"])
        print()


if __name__ == "__main__":
    main()
