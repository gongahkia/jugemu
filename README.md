# jugemu (tiny messages LLM + ChromaDB)

This is an intentionally **tiny and stupid** character-level language model trained on *your* message corpus, plus a **ChromaDB** vector store for retrieval.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If your default `python3` is 3.14+, use `python3.12` (ChromaDB native deps may not have 3.14 wheels yet).

## 1) Put your messages somewhere

Simplest format: a UTF-8 text file where each line is one message.

Example: `data/messages.txt`

## 2) Ingest to ChromaDB (vector DB)

```bash
cd /path/to/jugemu
python -m src.cli ingest \
  --messages data/messages.txt \
  --persist data/chroma \
  --collection messages
```

## 3) Train the tiny model

```bash
cd /path/to/jugemu
python -m src.cli train \
  --messages data/messages.txt \
  --out data/checkpoints \
  --epochs 5 \
  --batch-size 64 \
  --seq-len 256
```

## 4) Chat (retrieve similar messages + generate)

```bash
cd /path/to/jugemu
python -m src.cli chat \
  --persist data/chroma \
  --collection messages \
  --checkpoint data/checkpoints/latest.pt
```

You can also use the Makefile wrappers:

```bash
make ingest
make train
make chat
```

Notes:
- This model is deliberately small; it will not be factual or safe in general.
- It will mostly learn your *style* and short-range character patterns.
