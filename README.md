# Minimal Python RAG retrieval demo

A small demo of the retrieval stage of retrieval augmented generation (RAG).
It ranks a sample document collection using TF-IDF and cosine similarity and
prints the retrieved text, which could be passed to a language model as context.
This demo does not call a model or generate answers.

## Requirements

Python 3.9 or newer. Everything uses the standard library: no package installation,
API keys, network access, or external services are needed.

## Query the collection

From the repository directory:

```bash
python3 retrieval.py "tomatoes soil sunlight" --top-k 2
python3 retrieval.py "cosine similarity"
python3 retrieval.py --help
```

Results include the document title, ID, similarity score, and full text. Only
positive matches are returned; an empty or unknown query reports no matches.
The default maximum is three results. `--top-k` must be a positive integer.

Use your own collection with:

```bash
python3 retrieval.py "your query" --documents /path/to/documents.json
```

The JSON file must contain a list of objects with string `id`, `title`, and `text`
fields. See `documents.json` for five small examples. An empty collection is valid.
The default sample path is resolved relative to the script, so the CLI can also
run from another directory.

## How retrieval works

Text is lowercased and split into word tokens. Each document is represented by
raw term counts multiplied by smoothed inverse document frequency:
`idf(term) = log((1 + document_count) / (1 + document_frequency)) + 1`.
The query uses the same IDF weights and vocabulary. Cosine similarity compares
the query and document vectors; results are sorted from highest score to lowest,
with collection order breaking ties.

This is lexical retrieval: it has no stemming, stop-word removal, or semantic
embeddings. Unknown query terms are ignored. Scores measure vector similarity,
not probabilities or confidence in an answer.

## Run the tests

From the repository directory:

```bash
python3 -m unittest discover -s tests -v
```

Tests cover tokenization, TF-IDF weights, cosine similarity, ranking, result
limits, empty inputs, validation, and command-line behavior.
