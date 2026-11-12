"""Command-line entry points: `rsa ingest`, `rsa index`, `rsa search`."""

import json

import typer
from rich.console import Console
from rich.table import Table

from . import config

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command()
def ingest() -> None:
    """Chunk policy documents and load operational records into SQLite."""
    from .ingest import main

    console.print_json(json.dumps(main(), default=str))


@app.command()
def index() -> None:
    """Embed chunks (dense + BM25) and (re)build the Qdrant collection."""
    from .indexer import build_index

    console.print_json(json.dumps(build_index()))


@app.command()
def search(
    query: str,
    mode: str = typer.Option("full", help="dense | bm25 | hybrid | hybrid_rerank | full"),
    top_k: int = typer.Option(config.RERANK_TOP_K, help="Chunks to return"),
) -> None:
    """Run one query through the retrieval pipeline and show ranked chunks."""
    from .retriever import MODES, HybridRetriever

    # Validate at the CLI boundary so a typo gets a usage error, not a traceback
    # from deep inside the retriever.
    if mode not in MODES:
        raise typer.BadParameter(
            f"Unknown mode '{mode}'; expected one of: {', '.join(MODES)}",
            param_hint="'--mode'")

    try:
        retriever = HybridRetriever()
    except Exception as exc:  # noqa: BLE001 — missing collection, corrupt store, lock held
        # soft_wrap: keep the message on one line so it stays greppable in logs.
        Console(stderr=True).print(
            f"[red]Error:[/red] could not open the vector store at {config.QDRANT_PATH}: {exc}",
            soft_wrap=True)
        raise typer.Exit(1) from None

    result = retriever.retrieve(query, mode=mode, top_k=top_k)  # type: ignore[arg-type]

    table = Table(title=f"[{mode}] {query}", show_lines=True)
    table.add_column("#", justify="right")
    table.add_column("source / section")
    table.add_column("origin")
    table.add_column("score", justify="right")
    table.add_column("preview")
    for i, c in enumerate(result.chunks, 1):
        score = c.rerank_score if c.rerank_score is not None else c.score
        preview = " ".join(c.text.split())[:140]
        table.add_row(str(i), f"{c.source_file}#{c.chunk_index}\n{c.section_heading}",
                      c.origin, f"{score:.4f}", preview)
    console.print(table)
    console.print(f"timings (ms): {result.timings_ms}  total={result.total_ms}  "
                  f"reranked={result.candidates_reranked}")


@app.command()
def ask(
    question: str,
    customer: str = typer.Option(..., "--customer", "-c",
                                 help="Authenticated customer ID, e.g. CUST-1049"),
    show_trace: bool = typer.Option(False, "--trace", help="Print each step"),
) -> None:
    """Ask the support agent a question as a given customer (requires ANTHROPIC_API_KEY)."""
    from .agent import SupportAgent

    with SupportAgent(customer) as agent:
        result = agent.answer(question)

    if show_trace:
        table = Table(title="Execution trace", show_lines=False)
        table.add_column("#", justify="right")
        table.add_column("kind")
        table.add_column("name")
        table.add_column("ms", justify="right")
        table.add_column("ttft ms", justify="right")
        for s in result.steps:
            ttft = f"{s.ttft_ms:.0f}" if s.ttft_ms is not None else "-"
            table.add_row(str(s.index), s.kind, s.name, f"{s.latency_ms:.0f}", ttft)
        console.print(table)

    console.print(f"\n[bold]{result.answer}[/bold]\n")
    ttft_str = f"{result.ttft_ms:.0f}ms" if result.ttft_ms is not None else "n/a"
    console.print(f"outcome={result.outcome}  tools={result.tool_calls}  "
                  f"tokens={result.usage}  latency={result.latency_ms:.0f}ms")
    console.print(f"retrieval(chunks)={result.retrieval_ms:.0f}ms  "
                  f"ttft(final answer)={ttft_str}")
    if result.evidence:
        console.print("evidence: " + ", ".join(e["source"] for e in result.evidence))
    if result.cache is not None:
        console.print(f"semantic cache: {result.cache}")
    if result.usage.get("cached_input_tokens"):
        console.print(f"context cache: {result.usage['cached_input_tokens']} input tokens "
                      "read from the provider's prefix cache")
    console.print(f"trace: {result.trace_path}")


@app.command("eval-retrieval")
def eval_retrieval(
    modes: str = typer.Option("dense,bm25,hybrid,full", help="Comma-separated modes"),
    label: str = typer.Option("latest", help="Report name suffix"),
    k: int = typer.Option(config.RERANK_TOP_K, help="Cut-off for Hit@k / nDCG@k"),
) -> None:
    """Score retrieval modes against data/eval/retrieval_questions.json."""
    from .eval_retrieval import run

    run([m.strip() for m in modes.split(",") if m.strip()], label=label, k=k)


@app.command("eval-answers")
def eval_answers(
    label: str = typer.Option("latest", help="Report name suffix"),
    cases: str = typer.Option("", help="Comma-separated case ids (default: all)"),
) -> None:
    """Run the agent on data/eval/answer_cases.json; score correctness and faithfulness."""
    from .eval_answers import run

    run(label=label, case_ids=[c.strip() for c in cases.split(",") if c.strip()] or None)


@app.command("cost-report")
def cost_report(
    label: str = typer.Option("latest", help="Answer-eval report suffix to summarise"),
) -> None:
    """Token, latency and cost-sensitivity report from an eval run and its traces."""
    from .cost_report import run

    run(label)


@app.command("build-golden")
def build_golden(workers: int = typer.Option(4, help="Parallel judge calls")) -> None:
    """Build data/eval/golden_retrieval.json and golden_answers.json (pooled + LLM-graded)."""
    from .golden import build

    build(workers=workers)


@app.command("eval-golden-retrieval")
def eval_golden_retrieval(
    modes: str = typer.Option("dense,bm25,hybrid,hybrid_rerank,full", help="Comma-separated modes"),
    label: str = typer.Option("latest", help="Report name suffix"),
) -> None:
    """Precision/recall@k, MAP, MRR, nDCG on the chunk-labelled golden retrieval set."""
    from .eval_golden_retrieval import run

    run([m.strip() for m in modes.split(",") if m.strip()], label=label)


@app.command("eval-deepeval")
def eval_deepeval(
    label: str = typer.Option("latest", help="Report name suffix"),
    cases: str = typer.Option("", help="Comma-separated case ids (default: all)"),
    workers: int = typer.Option(4, help="Parallel cases during scoring"),
    reuse: bool = typer.Option(False, help="Reuse cached agent outputs for this label"),
) -> None:
    """Run the agent on the golden answer set; score with deepeval RAG metrics + G-Eval."""
    from .eval_deepeval import run

    run(label=label, case_ids=[c.strip() for c in cases.split(",") if c.strip()] or None,
        workers=workers, reuse=reuse)


@app.command("eval-ttft")
def eval_ttft(
    runs: int = typer.Option(2, help="Warm runs per question"),
    limit: int = typer.Option(0, help="Max questions (0 = all knowledge cases)"),
    customer: str = typer.Option("CUST-1001", help="Customer to run as"),
) -> None:
    """Time-to-first-token for RAG queries, broken down by stage."""
    from .eval_ttft import main

    main(runs=runs, limit=limit or None, customer=customer)


@app.command("calibrate-judge")
def calibrate_judge(label: str = typer.Option("latest", help="Report name suffix")) -> None:
    """Score the LLM judge against hand-labelled verdicts (data/eval/judge_calibration.json)."""
    from .judge_calibration import run

    run(label=label)


@app.command()
def chat(
    customer: str = typer.Option(..., "--customer", "-c", help="Signed-in customer ID"),
    session: str = typer.Option("", "--session", "-s",
                                help="Resume a session id (default: start a new one)"),
    memory: bool = typer.Option(True, "--memory/--no-memory",
                                help="Carry earlier turns + summary.md in the prompt"),
    cache: str = typer.Option("global", "--cache", help="Answer cache: off | session | global"),
) -> None:
    """Multi-turn chat with conversation memory. Commands: /summary  /turns  /exit."""
    from .session import ChatSession

    with ChatSession(customer, session or None, memory=memory, cache=cache) as s:
        console.print(f"session [bold]{s.session_id}[/bold] · memory={'on' if memory else 'off'}"
                      f" · cache={cache} · files: {s.dir}")
        while True:
            try:
                q = console.input("[bold cyan]you> [/bold cyan]").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            if q in ("/exit", "/quit"):
                break
            if q == "/summary":
                console.print(s.summary() or "(no summary yet)")
                continue
            if q == "/turns":
                console.print(f"{s.total_turns} turns, summarised up to "
                              f"{s.meta['summarized_upto']}, {len(s.turns())} on disk, "
                              f"{s.meta['irrelevant_turns_skipped']} irrelevant skipped")
                continue
            r = s.ask(q)
            console.print(f"[bold]assistant>[/bold] {r.answer}")
            console.print(f"[dim]outcome={r.outcome} tools={r.tool_calls} "
                          f"latency={r.latency_ms / 1000:.1f}s cache={r.cache} "
                          f"memory={r.memory}[/dim]")


@app.command("redteam-hallucination")
def redteam_hallucination(
    label: str = typer.Option("latest", help="Report name suffix"),
    cases: str = typer.Option("", help="Comma-separated case ids (default: all)"),
    rescore: bool = typer.Option(False, help="Re-score the saved report for this label "
                                             "(no agent or judge calls)"),
) -> None:
    """Adversarial prompts that bait fabrication; report the hallucination-free rate."""
    from .redteam import rescore as rescore_report
    from .redteam import run

    if rescore:
        rep = rescore_report(config.PROJECT_ROOT / "reports"
                             / f"redteam_hallucination_{label}.json")
        console.print_json(json.dumps(rep["summary"]))
        return
    run(label=label, case_ids=[c.strip() for c in cases.split(",") if c.strip()] or None)


@app.command("cache")
def cache_cmd(
    action: str = typer.Argument("stats", help="stats | clear"),
) -> None:
    """Inspect or clear the semantic answer cache (data/cache/semantic_cache.db)."""
    from .cache import SemanticCache

    cache = SemanticCache()
    if action == "clear":
        console.print(f"removed {cache.clear()} cached answers")
    elif action == "stats":
        console.print_json(json.dumps(cache.stats()))
    else:
        raise typer.BadParameter("expected 'stats' or 'clear'")


if __name__ == "__main__":
    app()
