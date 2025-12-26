# jugemu (tiny messages LLM + ChromaDB)

This is an intentionally **tiny and stupid** character-level language model trained on *your* message corpus, plus a **ChromaDB** vector store for retrieval.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Put your messages somewhere

Simplest format: a UTF-8 text file where each line is one message.

Example: `data/messages.txt`

## 2) Ingest to ChromaDB (vector DB)

```bash
python -m src.ingest_chroma \
  --messages data/messages.txt \
  --persist data/chroma \
  --collection messages
```

## 3) Train the tiny model

```bash
python -m src.train_char_model \
  --messages data/messages.txt \
  --out data/checkpoints \
  --epochs 5 \
  --batch-size 64 \
  --seq-len 256
```

## 4) Chat (retrieve similar messages + generate)

```bash
python -m src.chat \
  --persist data/chroma \
  --collection messages \
  --checkpoint data/checkpoints/latest.pt
```

Notes:
- This model is deliberately small; it will not be factual or safe in general.
- It will mostly learn your *style* and short-range character patterns.
