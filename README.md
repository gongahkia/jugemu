# `Jugemu`

A tiny message LLM.

Tiny, dumb, character-level language model trained only on your message corpus 

...

This is an intentionally **tiny and stupid** character-level language model trained on *your* message corpus, plus a **ChromaDB** vector store for retrieval.

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

...

## Stack

* ...
* ...
* ...

## Usage

> [!Important]  
> Please read the [legal disclaimer](#legal) before using `Jugemu`.

The below instructions are for locally hosting `Jugemu`. Additionally see [here](#configuration) for flags to augment `Jugemu`'s operations.

1. First run the below commands to setup `Jugemu`'s environment.

```console
$ git clone https://github.com/gongahkia/jugemu && cd jugemu
$ python3.12 -m venv .venv && source .venv/bin/activate
$ pip install -e '.[dev]'
```

2. Add your corpus to the relative path `./data/messages.txt` from the repository's root.
3. Next run these commands to initiate the [ingestion](#architecture) and [training](#architecture) cycle.

```console
$ jugemu browse --messages data/messages.txt --mode both --top 100 # browse corpus stats
$ jugemu ingest --messages data/messages.txt --persist data chroma --collection messages --fast-embedding-model --batch 32 # ingest the corpus to chromadb
$ jugemu train --messages data/messages.txt --out data/checkpoints --epochs 1 --steps-per-epoch 30 --batch-size 16 --seq-len 128 --log-every 10 # train a tiny checkpoint
```

4. Finally excecute the below to chat with `Jugemu`'s CLI chat context.

```console
$ jugemu chat --messages data/messages.txt --persist data/chroma --collection messages --checkpoint data/checkpoints/latest.pt --k 4 --max-new 180
```

5. Optionally run the below to export `Jugemu`'s CLI chat to a machine-readable `.json` format.

```console
$ jugemu chat --json --messages data/messages.txt --persist data/chroma --collection messages --checkpoint data/checkpoints/latest.pt
```

## Screenshots

![](./asset/reference/0.png)

![](./asset/reference/1.png)

![](./asset/reference/2.png)

![](./asset/reference/3.png)

![](./asset/reference/4.png)

![](./asset/reference/5.png)

## Architecture

> [!NOTE]  
> `Jugemu` is deliberately small and is therefore likely to not be factual or safe as it learns from short-range character patterns.

```mermaid

```

## Configuration

Most `Jugemu` commands additionally accept a global `--config` flag *(that otherwise defaults to `./config.toml` as below)*.

```toml
# ./config.toml

[paths]
messages = "data/messages.txt" # default messages path used by commands when you didn't pass --messages.
chroma_persist = "data/chroma" # chroma persist directory used by commands when you didn't pass --persist.
checkpoints = "data/checkpoints" # checkpoints directory used by train/chat/pipeline when you didn't pass --checkpoint/--out.

[chroma]
collection = "messages"

[embeddings]
model = "sentence-transformers/all-MiniLM-L6-v2"

[browse]
mode = "both" # chars|tokens|both
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

## Legal

...

## Reference

...