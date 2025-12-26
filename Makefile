.PHONY: help venv install ingest train chat clean

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# Defaults (override like: make ingest MESSAGES=/path/to/messages.txt)
MESSAGES ?= data/messages.txt
PERSIST ?= data/chroma
COLLECTION ?= messages
EMBED_MODEL ?= sentence-transformers/all-MiniLM-L6-v2

EPOCHS ?= 5
STEPS_PER_EPOCH ?= 500
BATCH_SIZE ?= 64
SEQ_LEN ?= 256
DEVICE ?= auto

MAX_NEW ?= 240
TEMPERATURE ?= 0.9
TOP_K ?= 60
K ?= 6

help:
	@echo "Targets:"
	@echo "  make venv      - create .venv using python3.12"
	@echo "  make install   - install deps into .venv"
	@echo "  make ingest    - embed+store messages into ChromaDB (via Rich CLI)"
	@echo "  make train     - train tiny char model and write checkpoints (via Rich CLI)"
	@echo "  make chat      - run retrieval+generate chat loop (via Rich CLI)"
	@echo "  make clean     - remove local artifacts (.venv, data/chroma, checkpoints)"
	@echo ""
	@echo "Common overrides:"
	@echo "  make ingest MESSAGES=/path/to/messages.txt"
	@echo "  make train EPOCHS=1 STEPS_PER_EPOCH=100 SEQ_LEN=128"
	@echo "  make chat CHECKPOINT=data/checkpoints/latest.pt"

venv:
	python3.12 -m venv $(VENV)

install: venv
	@$(PY) -m pip install -U pip
	@$(PIP) install -r requirements.txt

ingest: install
	@PYTHONPATH=$(PWD) $(PY) -m src.cli ingest \
		--messages $(MESSAGES) \
		--persist $(PERSIST) \
		--collection $(COLLECTION) \
		--embedding-model $(EMBED_MODEL)

train: install
	@PYTHONPATH=$(PWD) $(PY) -m src.cli train \
		--messages $(MESSAGES) \
		--out data/checkpoints \
		--epochs $(EPOCHS) \
		--steps-per-epoch $(STEPS_PER_EPOCH) \
		--batch-size $(BATCH_SIZE) \
		--seq-len $(SEQ_LEN) \
		--device $(DEVICE)

CHECKPOINT ?= data/checkpoints/latest.pt
chat: install
	@PYTHONPATH=$(PWD) $(PY) -m src.cli chat \
		--persist $(PERSIST) \
		--collection $(COLLECTION) \
		--embedding-model $(EMBED_MODEL) \
		--k $(K) \
		--checkpoint $(CHECKPOINT) \
		--device $(DEVICE) \
		--max-new $(MAX_NEW) \
		--temperature $(TEMPERATURE) \
		--top-k $(TOP_K)

clean:
	@rm -rf $(VENV) data/chroma data/checkpoints