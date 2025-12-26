from __future__ import annotations

import argparse

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

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


def _render_retrieval_table(retrieved: list[tuple[str, float]], k: int) -> Table:
    t = Table(title=f"Nearest Messages (k={k})", show_lines=False)
    t.add_column("Score", justify="right", no_wrap=True)
    t.add_column("Message")
    for text, score in retrieved:
        msg = text.replace("\n", " ").strip()
        if len(msg) > 140:
            msg = msg[:137] + "..."
        t.add_row(f"{score:.3f}", msg)
    return t


def _help_panel() -> Panel:
    body = Text(
        "Commands:\n"
        "  /help     Show this help\n"
        "  /exit     Quit\n"
        "  /clear    Clear screen\n"
        "\n"
        "Notes:\n"
        "- This is a tiny character model; expect nonsense.\n"
        "- Retrieval is from ChromaDB; the model just imitates style.\n"
    )
    return Panel(body, title="jugemu", border_style="cyan")


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
    ap.add_argument(
        "--show-retrieval",
        action="store_true",
        help="Print a table of retrieved similar messages each turn",
    )
    ap.add_argument(
        "--no-show-retrieval",
        action="store_true",
        help="Do not print retrieved messages table",
    )
    args = ap.parse_args()

    console = Console()
    show_retrieval = args.show_retrieval and not args.no_show_retrieval

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

    console.print(Panel("Type a message. Use /help for commands.", title="jugemu", border_style="cyan"))
    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye.")
            return

        if not user_text:
            continue

        if user_text in {"/exit", "/quit", ":q"}:
            console.print("Bye.")
            return
        if user_text == "/help":
            console.print(_help_panel())
            continue
        if user_text == "/clear":
            console.print("\n" * 80)
            continue

        hits = retrieve_similar(
            persist_dir=args.persist,
            collection_name=args.collection,
            query=user_text,
            k=args.k,
            embedding_model=args.embedding_model,
        )

        with console.status("Thinking…", spinner="dots"):
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

        if show_retrieval:
            console.print(_render_retrieval_table(retrieved, k=args.k))

        # Print only the tail after "YOU:" if possible.
        marker = "YOU:"
        if marker in out:
            answer = out.split(marker, 1)[-1].strip()
        else:
            answer = out.strip()

        console.print(Panel(answer, title="jugemu", border_style="green"))


if __name__ == "__main__":
    main()
