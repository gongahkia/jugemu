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
from .vector_store import VectorStore


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


def _strip_role_prefix(s: str) -> str:
    t = s.strip()
    lowered = t.lower()
    for prefix in ("you:", "user:"):
        if lowered.startswith(prefix):
            return t[len(prefix) :].lstrip()
    return t


def _clean_answer(s: str) -> str:
    # One-line, no role labels, no accidental next-turn start.
    t = s.replace("\r\n", "\n").replace("\r", "\n").strip()
    t = t.replace("\n", " ").strip()
    t = _strip_role_prefix(t)

    # If the model starts a new turn inline, truncate.
    lowered = t.lower()
    for marker in (" user:", " you:"):
        i = lowered.find(marker)
        if i != -1:
            t = t[:i].rstrip()
            break
    return t


def _choose_corpus_reply(
    *,
    retrieved: list[tuple[str, float, dict | None]],
    messages_path: Path,
    reply_strategy: str,
    min_score: float,
) -> str | None:
    if not messages_path.exists():
        return None

    best: tuple[float, str] | None = None
    for text, score, meta in retrieved:
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
        if not nxt:
            continue

        nxt_clean = _clean_answer(nxt)
        if not nxt_clean:
            continue

        # Avoid degenerate echo.
        if nxt_clean.strip() == _clean_answer(text).strip():
            continue

        if best is None or score > best[0]:
            best = (score, nxt_clean)

    return best[1] if best else None


def build_prompt(
    *,
    user_text: str,
    retrieved: list[tuple[str, float, dict | None]],
    messages_path: Path | None,
    template: str = "few-shot",
) -> str:
    """Build a prompt using only content from messages.txt (+ the user's input).

    Strategy:
    - For each retrieved message, look up the *next* message in messages.txt.
    - Build a few-shot pattern: msg -> next_msg.
    - Then append the user's message and let the model produce the next line.
    """
    template2 = str(template or "few-shot").strip().lower()
    if template2 in {"minimal", "min"}:
        return f"USER: {user_text}\nYOU: "

    if template2 in {"scaffold", "conversation-scaffold", "conversation"}:
        # Keep this extremely short; tiny models get confused by long instructions.
        return (
            "This is a chat. Reply as YOU. Keep it short.\n\n"
            f"USER: {user_text}\n"
            "YOU: "
        )

    if template2 not in {"few-shot", "fewshot", "retrieval-few-shot"}:
        raise ValueError("prompt template must be one of: minimal|scaffold|few-shot")

    blocks: list[str] = []
    if messages_path is not None and messages_path.exists():
        # Keep few-shot small; too many examples hurts tiny models.
        max_examples = 3
        for text, _score, meta in retrieved[: max_examples * 2]:
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
            # Use the same format as pair-training mode.
            u = _clean_answer(text)
            a = _clean_answer(nxt)
            if not u or not a:
                continue
            blocks.append(f"USER: {u}\nYOU: {a}")
            if len(blocks) >= max_examples:
                break

    # Keep prompt small and simple; examples first, then the user's message.
    if blocks:
        ctx = "\n\n".join(blocks)
        return f"{ctx}\n\nUSER: {user_text}\nYOU: "
    return f"USER: {user_text}\nYOU: "


def _render_retrieval_table(retrieved: list[tuple[str, float, dict | None]], k: int) -> Table:
    has_rerank = any((meta or {}).get("rerank_score") is not None for (_t, _s, meta) in retrieved)
    t = Table(title=f"Nearest Messages (k={k})", show_lines=False)
    t.add_column("Vec", justify="right", no_wrap=True)
    if has_rerank:
        t.add_column("Rerank", justify="right", no_wrap=True)
    t.add_column("Message")
    for text, score, meta in retrieved:
        msg = text.replace("\n", " ").strip()
        if len(msg) > 140:
            msg = msg[:137] + "..."
        cols = [f"{score:.3f}"]
        if has_rerank:
            rs = (meta or {}).get("rerank_score")
            cols.append(f"{float(rs):.3f}" if rs is not None else "")
        cols.append(msg)
        t.add_row(*cols)
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
        "--prompt-template",
        default="few-shot",
        choices=["few-shot", "scaffold", "minimal"],
        help="Prompt template used for generation (ignored for corpus replies).",
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
        "--stop-seq",
        action="append",
        default=[],
        help="Stop sequence (repeatable). Generation stops when any stop sequence appears.",
    )
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
    ap.add_argument(
        "--rerank",
        action="store_true",
        help="Rerank vector results with a cross-encoder (slower, often better).",
    )
    ap.add_argument(
        "--rerank-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder model name for reranking.",
    )
    ap.add_argument(
        "--rerank-top-k",
        type=int,
        default=20,
        help="How many vector hits to fetch before reranking.",
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
        prompt_template=args.prompt_template,
        checkpoint=args.checkpoint,
        device=args.device,
        max_new=args.max_new,
        temperature=args.temperature,
        top_k=args.top_k,
        stop_seq=list(args.stop_seq or []),
        show_retrieval=show_retrieval,
        rerank=bool(args.rerank),
        rerank_model=str(args.rerank_model),
        rerank_top_k=int(args.rerank_top_k),
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
    prompt_template: str = "few-shot",
    checkpoint: str,
    device: str = "auto",
    max_new: int = 240,
    temperature: float = 0.9,
    top_k: int = 60,
    stop_seq: list[str] | None = None,
    show_retrieval: bool = False,
    rerank: bool = False,
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    rerank_top_k: int = 20,
    store: VectorStore | None = None,
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
            store=store,
            rerank=bool(rerank),
            rerank_model=str(rerank_model),
            rerank_top_k=int(rerank_top_k),
        )

        with console.status("Thinking…", spinner="dots"):
            retrieved = [(h.text, h.score, h.metadata) for h in hits]
            corpus_reply: str | None = None
            if messages_path is not None and messages_path.exists():
                corpus_reply = _choose_corpus_reply(
                    retrieved=retrieved,
                    messages_path=messages_path,
                    reply_strategy=reply_strategy,
                    min_score=min_score,
                )

            if reply_strategy == "corpus" or (reply_strategy == "hybrid" and corpus_reply is not None):
                out = corpus_reply or ""
            else:
                prompt = build_prompt(
                    user_text=user_text,
                    retrieved=retrieved,
                    messages_path=messages_path,
                    template=prompt_template,
                )
                user_stops = [s for s in (stop_seq or []) if isinstance(s, str) and s]
                out = sample_text(
                    model=loaded.model,
                    prompt=prompt,
                    stoi=stoi,
                    itos=itos,
                    device=resolved_device,
                    max_new_tokens=max_new,
                    temperature=temperature,
                    top_k=top_k,
                    stop_on=["\n", "\nUSER:", "\nYOU:"] + user_stops,
                    return_full=False,
                )

        if show_retrieval:
            console.print(_render_retrieval_table(retrieved, k=k))

        answer = _clean_answer(out)
        console.print(Panel(answer, title="jugemu", border_style="green"))


if __name__ == "__main__":
    main()
