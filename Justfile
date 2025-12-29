# Optional shortcuts (requires `just`): https://github.com/casey/just
# Uses existing defaults from config.toml / CLI defaults.

set dotenv-load := false

messages := "data/messages.txt"
persist := "data/chroma"
collection := "messages"
embed_model := "sentence-transformers/all-MiniLM-L6-v2"

# Run embedding + store
ingest:
  PYTHONPATH=. python -m src.cli ingest --messages {{messages}} --persist {{persist}} --collection {{collection}} --embedding-model {{embed_model}}

# Train tiny char model
train:
  PYTHONPATH=. python -m src.cli train --messages {{messages}}

# Chat loop
chat:
  PYTHONPATH=. python -m src.cli chat --messages {{messages}} --persist {{persist}} --collection {{collection}} --embedding-model {{embed_model}}
