from __future__ import annotations

import argparse
from pathlib import Path

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .load_checkpoint import load_checkpoint, vocab_from_itos
from .retrieve import retrieve_similar
from .sample import sample_text


def _read_next_nonempty_line(path: Path, after_line_no: int) -> str | None:
    raw = path.read_text(encoding="utf-8", errors="replace")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = raw.split("\n")
    # after_line_no is 1-based.
    for ln in lines[after_line_no:]:
        msg = ln.strip()
        if msg:
            return msg
    return None


def build_prompt(
    *,
    user_text: str,
    retrieved: list[tuple[str, float, dict | None]],
    messages_path: Path | None,
) -> str:
    """Build a prompt using only content from messages.txt (+ the user's input).

    Strategy:
    - For each retrieved message, look up the *next* message in messages.txt.
    - Build a few-shot pattern: msg -> next_msg.
    - Then append the user's message and let the model produce the next line.
    """
    blocks: list[str] = []
    if messages_path is not None and messages_path.exists():
        for text, _score, meta in retrieved:
            if not meta:
                continue
            line_no = meta.get("line")
            if not isinstance(line_no, int):
                try:
                    line_no = int(line_no)
                except Exception:
                    continue
            nxt = _read_next_nonempty_line(messages_path, after_line_no=line_no)
            if not nxt:
                continue
            # No extra labels to avoid OOV prompt chars; just message \n response.
            blocks.append(f"{text}\n{nxt}")

    # Keep prompt small and simple; examples first, then the user's message.
    if blocks:
        ctx = "\n\n".join(blocks)
        return f"{ctx}\n\n{user_text}\n"
    # Fallback: just user message, then model continues.
    return f"{user_text}\n"


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
    ap.add_argument(
        "--messages",
        default=None,
        help="Optional: messages.txt path used to create msg->next-msg examples",
    )
    ap.add_argument(
        "--reply-strategy",
        default="hybrid",
        choices=["hybrid", "generate", "corpus"],
        help="hybrid: prefer corpus reply when confident, else generate",
    )
    ap.add_argument(
        "--min-score",
        type=float,
        default=0.35,
        help="Minimum retrieval score to use corpus reply in hybrid mode",
    )
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

    show_retrieval = args.show_retrieval and not args.no_show_retrieval
    run_chat(
        persist=args.persist,
        collection=args.collection,
        embedding_model=args.embedding_model,
        k=args.k,
        messages=args.messages,
        reply_strategy=args.reply_strategy,
        min_score=args.min_score,
        checkpoint=args.checkpoint,
        device=args.device,
        max_new=args.max_new,
        temperature=args.temperature,
        top_k=args.top_k,
        show_retrieval=show_retrieval,
    )


def run_chat(
    *,
    persist: str,
    collection: str = "messages",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    k: int = 6,
    messages: str | None = None,
    reply_strategy: str = "hybrid",
    min_score: float = 0.35,
    checkpoint: str,
    device: str = "auto",
    max_new: int = 240,
    temperature: float = 0.9,
    top_k: int = 60,
    show_retrieval: bool = False,
) -> None:
    console = Console()
    messages_path = Path(messages) if messages else None

    resolved_device = device
    if resolved_device == "auto":
        if torch.cuda.is_available():
            resolved_device = "cuda"
        elif torch.backends.mps.is_available():
            resolved_device = "mps"
        else:
            resolved_device = "cpu"

    loaded = load_checkpoint(checkpoint, device=resolved_device)
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
            persist_dir=persist,
            collection_name=collection,
            query=user_text,
            k=k,
            embedding_model=embedding_model,
        )

        with console.status("Thinking…", spinner="dots"):
            retrieved = [(h.text, h.score, h.metadata) for h in hits]
            corpus_reply: str | None = None
            if messages_path is not None and messages_path.exists():
                for _text, score, meta in retrieved:
                    if not meta:
                        continue
                    if reply_strategy == "hybrid" and score < min_score:
                        continue
                    line_no = meta.get("line")
                    if not isinstance(line_no, int):
                        try:
                            line_no = int(line_no)
                        except Exception:
                            continue
                    nxt = _read_next_nonempty_line(messages_path, after_line_no=line_no)
                    if nxt:
                        corpus_reply = nxt
                        break

            if reply_strategy == "corpus" or (reply_strategy == "hybrid" and corpus_reply is not None):
                out = corpus_reply or ""
            else:
                prompt = build_prompt(user_text=user_text, retrieved=retrieved, messages_path=messages_path)
                out = sample_text(
                    model=loaded.model,
                    prompt=prompt,
                    stoi=stoi,
                    itos=itos,
                    device=resolved_device,
                    max_new_tokens=max_new,
                    temperature=temperature,
                    top_k=top_k,
                    stop_on="\n",
                )

        if show_retrieval:
            console.print(_render_retrieval_table([(t, s) for (t, s, _) in retrieved], k=k))

        # For generated text, out may include prompt; show only the last line.
        answer = out.split("\n")[-1].strip()
        console.print(Panel(answer, title="jugemu", border_style="green"))


if __name__ == "__main__":
    main()
