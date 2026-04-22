"""
demo.py
-------
Interactive CLI for the Context-Aware Bangla RAG system.
Shows resolved query, retrieved hits, streaming answer,
and real-time per-stage latency breakdown on every turn.

Run:
    python demo.py
"""

import sys
import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.rule import Rule

from rag_pipeline import RAGPipeline, ConversationContext

# ── Config ───────────────────────────────────────────────────────────────────
TARGET_MS = 100.0
console   = Console()


# ── UI Helpers ────────────────────────────────────────────────────────────────

def print_header() -> None:
    console.print(Panel(
        "[bold cyan]Context-Aware Bangla RAG System[/bold cyan]\n"
        "[dim]Powered by gemma-3-1b (llama-cpp) + MiniLM-L12 + FAISS[/dim]\n\n"
        "[yellow]Commands:[/yellow]\n"
        "  Type any Bangla question and press Enter\n"
        "  [bold]reset[/bold]  — start a new conversation\n"
        "  [bold]quit[/bold]   — exit the demo",
        title="Bangla E-Commerce Assistant",
        border_style="cyan",
    ))


def print_turn_header(turn: int, raw: str, resolved: str) -> None:
    console.print(f"\n[bold cyan]── Turn {turn} ──────────────────────────────────[/bold cyan]")
    console.print(f"[dim]Raw query  :[/dim] [white]{raw}[/white]")
    if resolved != raw:
        console.print(f"[dim]Resolved   :[/dim] [green]{resolved}[/green] [dim](coref applied)[/dim]")
    else:
        console.print(f"[dim]Resolved   :[/dim] [white]{resolved}[/white]")


def print_hits(hits: list[dict]) -> None:
    table = Table(show_header=True, header_style="bold dim", box=None, padding=(0, 2))
    table.add_column("Rank", style="dim",    width=5)
    table.add_column("Product",              width=30)
    table.add_column("Price",  justify="right", style="green")
    table.add_column("Score",  justify="right", style="dim")

    for i, h in enumerate(hits, 1):
        score_str = "keyword" if h["score"] == 1.0 else f"{h['score']:.4f}"
        table.add_row(
            str(i),
            h["product_name"],
            f"{h['price_bdt']} টাকা",
            score_str,
        )

    console.print("\n[dim]Retrieved products:[/dim]")
    console.print(table)


def print_latency(lat: dict) -> None:
    total = lat["coref_ms"] + lat["embed_ms"] + lat["search_ms"] + lat["ttft_ms"]
    colour = "green" if total < TARGET_MS else "red"
    flag   = "✓ UNDER 100ms" if total < TARGET_MS else "✗ OVER 100ms"

    console.print(
        f"\n[dim]Latency :[/dim] "
        f"coref [cyan]{lat['coref_ms']:.1f}ms[/cyan]  "
        f"embed [cyan]{lat['embed_ms']:.1f}ms[/cyan]  "
        f"search [cyan]{lat['search_ms']:.1f}ms[/cyan]  "
        f"ttft [cyan]{lat['ttft_ms']:.1f}ms[/cyan]  "
        f"[bold {colour}]total {total:.1f}ms  {flag}[/bold {colour}]"
    )


def print_session_stats(history: list[dict]) -> None:
    if not history:
        return
    totals = [
        r["lat"]["coref_ms"] + r["lat"]["embed_ms"] +
        r["lat"]["search_ms"] + r["lat"]["ttft_ms"]
        for r in history
    ]
    pct_ok = sum(1 for t in totals if t < TARGET_MS) / len(totals) * 100
    console.print(Panel(
        f"[bold]Session Summary[/bold]\n\n"
        f"  Turns          : {len(history)}\n"
        f"  Mean total     : {sum(totals)/len(totals):.1f}ms\n"
        f"  % Under 100ms  : [{'green' if pct_ok==100 else 'yellow'}]{pct_ok:.0f}%[/]\n"
        f"  Coref resolved : {sum(1 for r in history if r['coref_applied'])} / {len(history)} turns",
        border_style="dim",
    ))


# ── Core Query Loop ───────────────────────────────────────────────────────────

def run_demo(pipeline: RAGPipeline) -> None:
    """Main interactive loop."""
    ctx:     ConversationContext = pipeline.new_context()
    history: list[dict]         = []
    turn:    int                 = 0

    print_header()
    console.print("\n[bold green]Pipeline ready. Start typing your question.[/bold green]\n")

    while True:
        # ── Prompt ───────────────────────────────────────────────────────────
        try:
            console.print(Rule(style="dim"))
            raw = console.input("[bold yellow]আপনি:[/bold yellow] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Exiting...[/dim]")
            break

        if not raw:
            continue

        if raw.lower() in {"quit", "exit", "q"}:
            print_session_stats(history)
            console.print("[dim]Goodbye.[/dim]")
            break

        if raw.lower() == "reset":
            ctx     = pipeline.new_context()
            history = []
            turn    = 0
            console.print("[yellow]Conversation reset.[/yellow]\n")
            continue

        # ── Query ─────────────────────────────────────────────────────────────
        turn += 1
        result         = pipeline.query(raw, ctx)
        resolved       = result["resolved_query"]
        hits           = result["hits"]
        answer         = result["answer"]
        lat            = result["latency"]
        coref_applied  = resolved != raw

        # ── Display ───────────────────────────────────────────────────────────
        print_turn_header(turn, raw, resolved)
        print_hits(hits)

        console.print(f"\n[bold green]Assistant:[/bold green] {answer}")
        print_latency(lat)

        # ── Log ───────────────────────────────────────────────────────────────
        history.append({
            "turn":          turn,
            "raw":           raw,
            "resolved":      resolved,
            "coref_applied": coref_applied,
            "answer":        answer,
            "lat":           lat,
        })


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    console.print("[bold]Initialising RAG pipeline...[/bold]")
    t0       = time.perf_counter()
    pipeline = RAGPipeline()
    console.print(f"[green]Ready in {(time.perf_counter()-t0)*1000:.0f}ms[/green]")

    console.print("[bold yellow]Warming up (3 passes)...[/bold yellow]")
    for _ in range(3):
        ctx = pipeline.new_context()
        pipeline.query("আপনাদের কোম্পানি কি নুডুলস বিক্রি করে?", ctx)
        pipeline.query("দাম কত টাকা?", ctx)
    console.print("[green]Warmup complete.[/green]\n")

    run_demo(pipeline)


if __name__ == "__main__":
    main()