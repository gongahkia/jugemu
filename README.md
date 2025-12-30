# `Jugemu`

A tiny message LLM.

This is an intentionally **tiny and stupid** character-level language model trained on *your* message corpus, plus a **ChromaDB** vector store for retrieval.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate

# Recommended (installs the `jugemu` CLI entrypoint)
pip install -e '.[dev]'

# Alternative (if you prefer requirements.txt)
# pip install -r requirements.txt
```

If your default `python3` is 3.14+, use `python3.12` (ChromaDB native deps may not have 3.14 wheels yet).

## Screenshot quickstart (steps 0–5)

These commands are optimized for fast demos + clean screenshots and use the literal path `data/messages.txt`.

### 0) Create a tiny demo corpus

```bash
mkdir -p data
cat > data/messages.txt <<'EOF'
hey are you free later?
yeah, what’s up?
thinking ramen tonight
down. same place as last time?
lol yes
ok see you 7ish
EOF
```

### 1) Browse corpus stats

```bash
jugemu browse --messages data/messages.txt --mode both --top 20
```

### 2) Ingest to ChromaDB (fast)

```bash
jugemu ingest \
  --messages data/messages.txt \
  --persist data/chroma \
  --collection messages \
  --fast-embedding-model \
  --batch 32
```

### 3) Train a tiny checkpoint (fast)

```bash
jugemu train \
  --messages data/messages.txt \
  --out data/checkpoints \
  --epochs 1 \
  --steps-per-epoch 30 \
  --batch-size 16 \
  --seq-len 128 \
  --log-every 10
```

### 4) Chat (retrieval + generation)

```bash
jugemu chat \
  --messages data/messages.txt \
  --persist data/chroma \
  --collection messages \
  --checkpoint data/checkpoints/latest.pt \
  --k 4 \
  --max-new 180
```

### 5) Optional: machine-readable chat JSON

```bash
jugemu chat \
  --json \
  --messages data/messages.txt \
  --persist data/chroma \
  --collection messages \
  --checkpoint data/checkpoints/latest.pt
```

## 1) Put your messages somewhere

Simplest format: a UTF-8 text file where each line is one message.

Example: `data/messages.txt`

## 2) Ingest to ChromaDB (vector DB)

```bash
cd /path/to/jugemu
jugemu ingest \
  --messages data/messages.txt \
  --persist data/chroma \
  --collection messages
```

Privacy notes:
- jugemu runs **locally**. It does not call remote LLM APIs.
- Ingestion writes a local ChromaDB database under `--persist` (default: `data/chroma`).
- Embeddings are computed locally via `sentence-transformers` and stored in that ChromaDB directory.
- If you want to scrub common PII-like strings before they ever hit the vector DB, use `--redact`.

Example (redact emails + phone numbers before embedding/storing):

```bash
jugemu ingest \
  --messages data/messages.txt \
  --persist data/chroma \
  --collection messages \
  --redact \
  --redact-type email \
  --redact-type phone
```

## 3) Train the tiny model

```bash
cd /path/to/jugemu
jugemu train \
  --messages data/messages.txt \
  --out data/checkpoints \
  --epochs 5 \
  --batch-size 64 \
  --seq-len 256
```

Optional (usually more coherent chat-style replies):

```bash
jugemu train \
  --messages data/messages.txt \
  --out data/checkpoints \
  --training-mode pairs
```

Privacy notes:
- Training writes checkpoints under `--out` (default: `data/checkpoints`).
- A small `run.json` (config + git hash) and `meta.json` are written alongside checkpoints.
- If you want to scrub common PII-like strings before training, use `--redact`.

## 4) Chat (retrieve similar messages + generate)

```bash
cd /path/to/jugemu
jugemu chat \
  --messages data/messages.txt \
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

## Configuration (config.toml)

Most commands accept a global `--config` option (defaults to `./config.toml` when present).

Supported keys:

```toml
[paths]
# Default messages path used by commands when you didn't pass --messages.
messages = "data/messages.txt"

# Chroma persist directory used by commands when you didn't pass --persist.
chroma_persist = "data/chroma"

# Checkpoints directory used by train/chat/pipeline when you didn't pass --checkpoint/--out.
checkpoints = "data/checkpoints"

[chroma]
collection = "messages"

[embeddings]
model = "sentence-transformers/all-MiniLM-L6-v2"

[browse]
mode = "both"       # chars|tokens|both
top = 50
min_count = 1
json = false

[export_retrieval]
samples = 10
k = 6
seed = 1337
embed_batch_size = 32
out_format = "jsonl" # json|jsonl
no_print = false

[pipeline]
vector_backend = "chroma" # chroma|cassandra
smoke_prompt = "hello"
smoke_max_new = 120

[chat]
k = 6
max_new = 240
temperature = 0.9
top_k = 60
device = "auto" # auto/cpu/mps/cuda
```
