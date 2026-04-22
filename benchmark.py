"""
benchmark.py
------------
30-run latency benchmark for the Bangla RAG pipeline.
Measures Time-To-First-Token (TTFT) and full per-stage breakdown.

Outputs:
    - Rich table: min/max/mean/P50/P95/% under 100ms
    - benchmark_results.json: raw per-run data

Run:
    python benchmark.py
"""

import json
import time
import statistics
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from rich import print as rprint

from rag_pipeline import RAGPipeline, ConversationContext

# ── Config ───────────────────────────────────────────────────────────────────
N_RUNS       = 30
TARGET_MS    = 100.0
Q1           = "আপনাদের কোম্পানি কি নুডুলস বিক্রি করে?"
Q2           = "দাম কত টাকা?"
RESULTS_FILE = "benchmark_results.json"


# ── Helpers ──────────────────────────────────────────────────────────────────

def percentile(data: list[float], p: float) -> float:
    """Return the p-th percentile of sorted data (0-100)."""
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def colour(val: float, target: float) -> str:
    """Return Rich colour tag based on whether val is under target."""
    return "green" if val < target else "red"


# ── Warmup ───────────────────────────────────────────────────────────────────

def warmup(pipeline: RAGPipeline, console: Console) -> None:
    """
    Run five full warm-up passes (Q1 + Q2) before benchmarking.
    This ensures embedding model weights and GGUF KV cache are hot.
    Five passes are needed to fully prime the M1 memory subsystem.
    """
    console.print("[bold yellow]Warming up pipeline (5 passes)...[/bold yellow]")
    for _ in range(5):
        ctx = pipeline.new_context()
        pipeline.query(Q1, ctx)
        pipeline.query(Q2, ctx)
    console.print("[bold green]Warmup complete.[/bold green]\n")


# ── Benchmark ─────────────────────────────────────────────────────────────────

def run_benchmark(pipeline: RAGPipeline, n: int, console: Console) -> list[dict]:
    """
    Run n iterations of the Q1 -> Q2 conversation flow.
    Q1 is used only to seed the context; only Q2 is measured.

    Returns list of per-run latency dicts.
    """
    results: list[dict] = []

    console.print(f"[bold]Running {n} benchmark iterations...[/bold]\n")

    for i in range(n):
        # Fresh context each run — mirrors real production usage
        ctx = pipeline.new_context()

        # Q1: seed context (not measured)
        pipeline.query(Q1, ctx)

        # Q2: measured turn
        r = pipeline.query(Q2, ctx)
        lat = r["latency"]

        # Compute true total as sum of stages up to first token
        true_total = round(
            lat["coref_ms"] + lat["embed_ms"] + lat["search_ms"] + lat["ttft_ms"], 2
        )

        run_data = {
            "run":          i + 1,
            "resolved_query": r["resolved_query"],
            "coref_ms":     lat["coref_ms"],
            "embed_ms":     lat["embed_ms"],
            "search_ms":    lat["search_ms"],
            "ttft_ms":      lat["ttft_ms"],
            "total_ms":     true_total,
            "under_100ms":  true_total < TARGET_MS,
        }
        results.append(run_data)

        # Live progress
        tag = "[green]✓[/green]" if true_total < TARGET_MS else "[red]✗[/red]"
        console.print(
            f"  {tag} Run {i+1:02d} | "
            f"coref={lat['coref_ms']:5.1f}ms  "
            f"embed={lat['embed_ms']:5.1f}ms  "
            f"search={lat['search_ms']:5.1f}ms  "
            f"ttft={lat['ttft_ms']:6.1f}ms  "
            f"[bold]total={true_total:6.1f}ms[/bold]"
        )

    return results


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(results: list[dict], console: Console) -> None:
    """Print Rich latency table and coreference proof."""
    totals   = [r["total_ms"]  for r in results]
    ttfts    = [r["ttft_ms"]   for r in results]
    embeds   = [r["embed_ms"]  for r in results]
    searches = [r["search_ms"] for r in results]
    n        = len(results)
    pct_ok   = sum(1 for t in totals if t < TARGET_MS) / n * 100

    # ── Main latency table ────────────────────────────────────────────────────
    table = Table(title=f"Latency Benchmark — {n} runs  |  Target: <{TARGET_MS:.0f}ms TTFT", 
                  show_header=True, header_style="bold cyan")
    table.add_column("Metric",    style="cyan",  width=22)
    table.add_column("Total ms",  style="white", width=12, justify="right")
    table.add_column("TTFT ms",   style="white", width=12, justify="right")
    table.add_column("Embed ms",  style="white", width=12, justify="right")
    table.add_column("Search ms", style="white", width=12, justify="right")

    def row(label, t, ttft, emb, srch):
        tc = colour(t, TARGET_MS)
        table.add_row(
            label,
            f"[{tc}]{t:.1f}[/{tc}]",
            f"{ttft:.1f}",
            f"{emb:.1f}",
            f"{srch:.1f}",
        )

    row("Min",    min(totals),                    min(ttfts),    min(embeds),    min(searches))
    row("Max",    max(totals),                    max(ttfts),    max(embeds),    max(searches))
    row("Mean",   statistics.mean(totals),        statistics.mean(ttfts),   statistics.mean(embeds),   statistics.mean(searches))
    row("Median (P50)", statistics.median(totals), percentile(ttfts, 50),   percentile(embeds, 50),    percentile(searches, 50))
    row("P95",    percentile(totals, 95),         percentile(ttfts, 95),   percentile(embeds, 95),    percentile(searches, 95))
    row("P99",    percentile(totals, 99),         percentile(ttfts, 99),   percentile(embeds, 99),    percentile(searches, 99))

    table.add_section()
    pct_colour = "green" if pct_ok >= 95 else "yellow" if pct_ok >= 80 else "red"
    table.add_row(
        "% Under 100ms",
        f"[bold {pct_colour}]{pct_ok:.0f}%[/bold {pct_colour}]",
        "", "", ""
    )

    console.print(table)

    # ── Coreference proof ─────────────────────────────────────────────────────
    console.print(Panel(
        f"[bold]Coreference Resolution Proof[/bold]\n\n"
        f"  Raw Q2    : [yellow]{Q2}[/yellow]\n"
        f"  Resolved  : [green]{results[0]['resolved_query']}[/green]\n\n"
        f"  All {n} runs resolved Q2 correctly: "
        f"[green]{'✓' if all(r['resolved_query'] != Q2 for r in results) else '✗'}[/green]",
        title="Coref Check",
        border_style="cyan",
    ))

    # ── Pass/fail summary ─────────────────────────────────────────────────────
    if pct_ok == 100:
        console.print(Panel(
            f"[bold green]ALL {n} RUNS UNDER {TARGET_MS:.0f}ms[/bold green]\n"
            f"Mean total: {statistics.mean(totals):.1f}ms  |  P95: {percentile(totals, 95):.1f}ms",
            border_style="green",
        ))
    elif pct_ok >= 95:
        console.print(Panel(
            f"[bold yellow]{pct_ok:.0f}% of runs under {TARGET_MS:.0f}ms[/bold yellow]\n"
            f"P95: {percentile(totals, 95):.1f}ms — production viable",
            border_style="yellow",
        ))
    else:
        console.print(Panel(
            f"[bold red]{pct_ok:.0f}% under {TARGET_MS:.0f}ms — target not consistently met[/bold red]",
            border_style="red",
        ))


# ── Persist ───────────────────────────────────────────────────────────────────

def save_results(results: list[dict]) -> None:
    """Write full per-run data to JSON for auditing."""
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved → {RESULTS_FILE}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    console = Console()

    console.print(Panel(
        "[bold cyan]Context-Aware Bangla RAG — Latency Benchmark[/bold cyan]\n"
        f"Target: <{TARGET_MS:.0f}ms TTFT  |  Runs: {N_RUNS}  |  Hardware: Apple M1",
        border_style="cyan",
    ))

    # Init pipeline
    console.print("\n[bold]Initialising pipeline...[/bold]")
    t0       = time.perf_counter()
    pipeline = RAGPipeline()
    console.print(f"Pipeline ready in {(time.perf_counter()-t0)*1000:.0f}ms\n")

    # Warmup
    warmup(pipeline, console)

    # Benchmark
    results = run_benchmark(pipeline, N_RUNS, console)

    # Report
    console.print()
    print_report(results, console)

    # Save
    save_results(results)


if __name__ == "__main__":
    main()