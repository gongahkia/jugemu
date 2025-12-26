from __future__ import annotations

import argparse

import torch

from .load_checkpoint import load_checkpoint, vocab_from_itos
from .retrieve import retrieve_similar
from .sample import sample_text


def build_prompt(user_text: str, retrieved: list[tuple[str, float]]) -> str:
    # Keep it extremely simple: prepend similar past messages.
    ctx = "\n".join([f"- ({score:.2f}) {text}" for text, score in retrieved])
    return (
        "PAST MESSAGES (similar):\n"
        f"{ctx}\n\n"
        "USER: "
        f"{user_text}\n"
        "YOU: "
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--persist", required=True, help="ChromaDB persistence directory")
    ap.add_argument("--collection", default="messages")
    ap.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--checkpoint", required=True, help="Path to model checkpoint .pt")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--max-new", type=int, default=240)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=60)
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    loaded = load_checkpoint(args.checkpoint, device=device)
    stoi, itos = vocab_from_itos(loaded.vocab_itos)

    print("Type a message. Ctrl+C to exit.")
    while True:
        user_text = input("> ").strip()
        if not user_text:
            continue

        hits = retrieve_similar(
            persist_dir=args.persist,
            collection_name=args.collection,
            query=user_text,
            k=args.k,
            embedding_model=args.embedding_model,
        )
        retrieved = [(h.text, h.score) for h in hits]

        prompt = build_prompt(user_text, retrieved)
        out = sample_text(
            model=loaded.model,
            prompt=prompt,
            stoi=stoi,
            itos=itos,
            device=device,
            max_new_tokens=args.max_new,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        # Print only the tail after "YOU:" if possible.
        marker = "YOU:"
        if marker in out:
            print(out.split(marker, 1)[-1].strip())
        else:
            print(out.strip())


if __name__ == "__main__":
    main()
