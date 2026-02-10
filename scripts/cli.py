"""
OpenCausality Platform — Unified CLI.

Usage:
    opencausality list-studies
    opencausality example [study]
    opencausality dag run <path>
    opencausality query --dag <path>
    opencausality init
    opencausality config show
"""

import os
import re
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


def _find_project_root() -> Path:
    """Walk up from this file to find the directory containing pyproject.toml."""
    anchor = Path(__file__).resolve().parent  # scripts/
    for candidate in (anchor.parent, anchor, *anchor.parent.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd()


PROJECT_ROOT = _find_project_root()


def resolve_path(p: str | Path) -> Path:
    """Resolve a path: if absolute or exists as-is, use it; otherwise resolve
    relative to the project root so the CLI works from any working directory."""
    p = Path(p)
    if p.is_absolute() or p.exists():
        return p
    rooted = PROJECT_ROOT / p
    if rooted.exists():
        return rooted
    return p  # fall back to original for error messages

BANNER = r"""[dim]    ___                    ____                        _ _ _
   / _ \ _ __   ___ _ __  / ___|__ _ _   _ ___  __ _ | (_) |_ _   _
  | | | | '_ \ / _ \ '_ \| |   / _` | | | / __|/ _` || | | __| | | |
  | |_| | |_) |  __/ | | | |__| (_| | |_| \__ \ (_| || | | |_| |_| |
   \___/| .__/ \___|_| |_|\____\__,_|\__,_|___/\__,_||_|_|\__|\__, |
        |_|                                                    |___/[/dim]

  [italic]sketch your causal story ~ we find the papers, collect the
  data, pick the estimator, run the regressions, and tell you
  what's wrong before you publish something you'll regret[/italic]
"""

app = typer.Typer(
    name="opencausality",
    help="OpenCausality — Open-Source Causal Inference Platform",
    no_args_is_help=True,
)
dag_app = typer.Typer(
    name="dag",
    help="DAG-based agentic estimation commands",
)
config_app = typer.Typer(
    name="config",
    help="Configuration management commands",
)
data_app = typer.Typer(
    name="data",
    help="Data ingestion and management commands",
)
agent_app = typer.Typer(
    name="agent",
    help="Agent management commands (DataScout, ModelSmith, Estimator, Judge)",
)
loop_app = typer.Typer(
    name="loop",
    help="Estimation loop control (start/stop/status/once/log)",
)
benchmark_app = typer.Typer(
    name="benchmark",
    help="Benchmarking and evaluation commands",
)
discover_app = typer.Typer(
    name="discover",
    help="Causal discovery commands (PC, GES, NOTEARS)",
)
app.add_typer(dag_app, name="dag")
app.add_typer(config_app, name="config")
app.add_typer(data_app, name="data")
app.add_typer(agent_app, name="agent")
app.add_typer(loop_app, name="loop")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(discover_app, name="discover")


@app.callback()
def main_callback(ctx: typer.Context):
    """Auto-ingest new data files on every command."""
    # Skip auto-ingest for data subcommands (they handle it explicitly)
    # and for help/completion commands
    cmd_name = ctx.invoked_subcommand
    if cmd_name not in ("data",):
        try:
            from shared.engine.ingest import auto_ingest
            auto_ingest(quiet=True)
        except Exception:
            pass  # Never block CLI on ingest failure

console = Console()


# ============================================================================
# Built-in Example Studies
# ============================================================================

EXAMPLE_STUDIES = {
    "welfare": {
        "module": "studies.household_welfare.src.cli",
        "name": "Household Welfare",
        "description": "Oil price shocks -> household income",
    },
    "credit": {
        "module": "studies.credit_default.src.cli",
        "name": "Credit Default",
        "description": "Income changes -> credit default risk",
    },
    "passthrough": {
        "module": "studies.fx_passthrough.src.cli",
        "name": "FX Passthrough",
        "description": "FX -> inflation -> income -> expenditure",
    },
}


@app.command("example")
def example_study(
    study: Optional[str] = typer.Argument(None, help="Study name: welfare, credit, passthrough"),
):
    """Run a built-in example study."""
    import importlib

    selected = study

    if selected is None:
        # Interactive picker
        console.print(BANNER)
        console.print("[bold cyan]Available Example Studies[/bold cyan]\n")
        for i, (key, info) in enumerate(EXAMPLE_STUDIES.items(), 1):
            console.print(f"  [green]{i}[/green]. [cyan]{info['name']}[/cyan] — {info['description']}")
        console.print()
        choice = typer.prompt("Select a study (number or name)", default="1")

        # Resolve choice
        if choice.isdigit():
            idx = int(choice) - 1
            keys = list(EXAMPLE_STUDIES.keys())
            if 0 <= idx < len(keys):
                selected = keys[idx]
        if selected is None or selected not in EXAMPLE_STUDIES:
            # Try fuzzy match on name
            for key in EXAMPLE_STUDIES:
                if choice.lower() in key:
                    selected = key
                    break
        if selected is None or selected not in EXAMPLE_STUDIES:
            console.print(f"[red]Unknown study: {choice}[/red]")
            raise typer.Exit(1)

    if selected not in EXAMPLE_STUDIES:
        console.print(f"[red]Unknown study: {selected}. Choose from: {', '.join(EXAMPLE_STUDIES)}[/red]")
        raise typer.Exit(1)

    info = EXAMPLE_STUDIES[selected]
    try:
        mod = importlib.import_module(info["module"])
        console.print(f"\n[bold cyan]Entering: {info['name']}[/bold cyan]\n")
        mod.app(standalone_mode=False)
    except ImportError:
        console.print(f"[red]Study module not found: {info['module']}[/red]")
        raise typer.Exit(1)


@app.command("list-studies")
def list_studies():
    """List all available research studies."""
    table = Table(title="OpenCausality Research Studies")

    table.add_column("Study", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Status", style="green")
    table.add_column("CLI", style="yellow")

    for key, info in EXAMPLE_STUDIES.items():
        table.add_row(
            key,
            info["description"],
            "Active",
            f"opencausality example {key}",
        )

    console.print(table)


@app.command("info")
def info():
    """Show information about the research platform."""
    console.print(BANNER)
    console.print("Version: 0.3.0")
    console.print("\n[bold]Example Studies:[/bold]")
    for i, (key, study_info) in enumerate(EXAMPLE_STUDIES.items(), 1):
        console.print(f"  {i}. [cyan]{key}[/cyan] - {study_info['description']}")
    console.print()
    console.print("[bold]Shared Infrastructure:[/bold]")
    console.print("  - shared/data/ - Data clients (BNS, FRED, exchange rate, CPI)")
    console.print("  - shared/model/ - Inference, event study, small-N methods")
    console.print("  - shared/engine/ - Scenario simulation base")
    console.print("  - shared/agentic/ - DAG-based agentic estimation")
    console.print("  - shared/llm/ - LLM abstraction layer\n")
    console.print("[bold]Usage:[/bold]")
    console.print("  opencausality list-studies       List all studies")
    console.print("  opencausality example            Interactive study picker")
    console.print("  opencausality example welfare    Run welfare study directly")
    console.print("  opencausality dag run <path>     Run a DAG specification")
    console.print("  opencausality query              Interactive query REPL")
    console.print("  opencausality init               Setup wizard + new study scaffold")
    console.print("  opencausality config show        Show configuration\n")


# ============================================================================
# Init Command
# ============================================================================

@app.command("init")
def init_project():
    """Interactive setup wizard: create .env with API keys."""
    env_path = Path(".env")
    if env_path.exists():
        overwrite = typer.confirm(".env already exists. Overwrite?", default=False)
        if not overwrite:
            console.print("[dim]Aborted.[/dim]")
            raise typer.Exit(0)

    console.print(BANNER)
    console.print(Panel("[bold cyan]OpenCausality Setup[/bold cyan]", subtitle="Creates .env"))
    console.print()

    # Collect keys interactively
    anthropic_key = typer.prompt(
        "Anthropic API key (for query REPL)", default="", show_default=False,
    )
    fred_key = typer.prompt(
        "FRED API key (https://fred.stlouisfed.org/docs/api/api_key.html)",
        default="", show_default=False,
    )
    s2_key = typer.prompt(
        "Semantic Scholar API key (optional)", default="", show_default=False,
    )

    llm_provider = typer.prompt("LLM provider", default="anthropic")
    llm_model = typer.prompt("LLM model", default="claude-sonnet-4-5-20250929")
    dag_path = typer.prompt("Default DAG path", default="config/agentic/dags/kspi_k2_full.yaml")

    # Validate FRED key format if provided
    if fred_key and len(fred_key) < 10:
        console.print("[yellow]Warning: FRED API key looks too short.[/yellow]")

    # Write .env
    lines = [
        "# OpenCausality Platform",
        f"ANTHROPIC_API_KEY={anthropic_key}",
        f"LLM_PROVIDER={llm_provider}",
        f"LLM_MODEL={llm_model}",
        "",
        f"FRED_API_KEY={fred_key}",
        f"SEMANTIC_SCHOLAR_API_KEY={s2_key}",
        "OPENALEX_MAILTO=",
        "UNPAYWALL_EMAIL=",
        "CORE_API_KEY=",
        "",
        "CACHE_DIR=.cache",
        "DATA_DIR=data",
        "OUTPUT_DIR=outputs",
        "",
        f"DEFAULT_DAG_PATH={dag_path}",
        "DEFAULT_QUERY_MODE=REDUCED_FORM",
        "LOG_LEVEL=INFO",
    ]
    env_path.write_text("\n".join(lines) + "\n")
    console.print(f"\n[green]Wrote {env_path}[/green]")
    console.print("[dim]Run 'opencausality config doctor' to verify.[/dim]")

    create_study = typer.confirm("\nCreate a new study?", default=False)
    if create_study:
        _scaffold_new_study()


def _scaffold_new_study() -> None:
    """Interactively scaffold a new study directory."""
    name = typer.prompt("Study name (snake_case, e.g. trade_balance)")
    description = typer.prompt("Short description", default=f"{name} causal study")

    # Sanitize name
    slug = re.sub(r"[^a-z0-9_]", "_", name.lower()).strip("_")
    study_dir = PROJECT_ROOT / "studies" / slug

    if study_dir.exists():
        console.print(f"[yellow]Directory already exists: {study_dir}[/yellow]")
        return

    # Create directory structure
    (study_dir / "src").mkdir(parents=True)
    (study_dir / "config").mkdir()
    (study_dir / "data").mkdir()
    (study_dir / "outputs").mkdir()

    # __init__.py
    (study_dir / "src" / "__init__.py").write_text("")

    # cli.py stub
    cli_content = f'''"""CLI for {slug} study."""

import typer
from rich.console import Console

app = typer.Typer(help="{description}")
console = Console()


@app.command("fetch-data")
def fetch_data():
    """Fetch data for this study."""
    console.print("[dim]TODO: implement data fetching[/dim]")


@app.command("estimate")
def estimate():
    """Run estimation pipeline."""
    console.print("[dim]TODO: implement estimation[/dim]")


@app.command("simulate")
def simulate():
    """Run scenario simulation."""
    console.print("[dim]TODO: implement simulation[/dim]")
'''
    (study_dir / "src" / "cli.py").write_text(cli_content)

    # dag.yaml template
    dag_content = f"""# {description}
# DAG specification — see INVARIANTS.md for schema rules

metadata:
  name: {slug}
  description: "{description}"
  target_node: outcome
  frequency: M

nodes:
  - id: treatment
    name: Treatment Variable
    type: observed
    frequency: M

  - id: outcome
    name: Outcome Variable
    type: observed
    frequency: M

edges:
  - id: treatment_to_outcome
    from: treatment
    to: outcome
    type: directed
    lag: 0
"""
    (study_dir / "config" / "dag.yaml").write_text(dag_content)

    console.print(f"\n[green]Scaffolded new study:[/green] {study_dir}")
    console.print(f"  src/cli.py       — Typer app with fetch-data, estimate, simulate stubs")
    console.print(f"  config/dag.yaml  — Minimal DAG template")
    console.print(f"  data/            — Data directory")
    console.print(f"  outputs/         — Output directory")
    console.print(f"\n[dim]Run 'opencausality example {slug}' after registering in EXAMPLE_STUDIES.[/dim]")


# ============================================================================
# Config Commands
# ============================================================================

def _mask_key(value: str) -> str:
    """Mask an API key for display — NEVER print full secrets."""
    if not value:
        return "[dim]<not set>[/dim]"
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + "*" * (len(value) - 8) + value[-4:]


@config_app.command("show")
def config_show():
    """Show current configuration (API keys masked)."""
    from config.settings import get_settings

    settings = get_settings()

    table = Table(title="OpenCausality Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("llm_provider", settings.llm_provider)
    table.add_row("llm_model", settings.llm_model)
    table.add_row("anthropic_api_key", _mask_key(settings.anthropic_api_key))
    table.add_row("fred_api_key", _mask_key(settings.fred_api_key))
    table.add_row("semantic_scholar_api_key", _mask_key(settings.semantic_scholar_api_key))
    table.add_row("openalex_mailto", settings.openalex_mailto or "[dim]<not set>[/dim]")
    table.add_row("unpaywall_email", settings.unpaywall_email or "[dim]<not set>[/dim]")
    table.add_row("core_api_key", _mask_key(settings.core_api_key))
    table.add_row("", "")
    table.add_row("default_dag_path", settings.default_dag_path)
    table.add_row("default_query_mode", settings.default_query_mode)
    table.add_row("cache_dir", str(settings.cache_dir))
    table.add_row("data_dir", str(settings.data_dir))
    table.add_row("output_dir", str(settings.output_dir))
    table.add_row("log_level", settings.log_level)

    console.print(table)


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Configuration key (e.g. ANTHROPIC_API_KEY)"),
    value: str = typer.Argument(..., help="New value"),
):
    """Set a configuration value in .env."""
    env_path = Path(".env")
    if not env_path.exists():
        console.print("[red].env not found. Run 'opencausality init' first.[/red]")
        raise typer.Exit(1)

    content = env_path.read_text()
    key_upper = key.upper()

    # Replace existing line or append
    pattern = re.compile(rf"^{re.escape(key_upper)}=.*$", re.MULTILINE)
    if pattern.search(content):
        content = pattern.sub(f"{key_upper}={value}", content)
    else:
        content = content.rstrip("\n") + f"\n{key_upper}={value}\n"

    env_path.write_text(content)
    console.print(f"[green]Set {key_upper} in .env[/green]")


@config_app.command("doctor")
def config_doctor():
    """Diagnose configuration: check API keys, files, connectivity."""
    from config.settings import get_settings

    settings = get_settings()
    issues: list[str] = []
    ok: list[str] = []

    console.print(Panel("[bold cyan]Configuration Doctor[/bold cyan]"))

    # Check API keys
    if settings.anthropic_api_key:
        ok.append("Anthropic API key is set")
    else:
        issues.append("ANTHROPIC_API_KEY not set (query REPL will use regex fallback)")

    if settings.fred_api_key:
        ok.append("FRED API key is set")
    else:
        issues.append("FRED_API_KEY not set (data fetching will fail)")

    if settings.semantic_scholar_api_key:
        ok.append("Semantic Scholar API key is set")
    else:
        issues.append("SEMANTIC_SCHOLAR_API_KEY not set (literature search may be rate-limited)")

    # Check default DAG file
    dag_path = resolve_path(settings.default_dag_path)
    if dag_path.exists():
        ok.append(f"Default DAG exists: {dag_path}")
    else:
        issues.append(f"Default DAG not found: {dag_path}")

    # Check cards directory
    cards_dir = Path(settings.output_dir) / "agentic" / "edge_cards"
    if cards_dir.exists():
        card_count = len(list(cards_dir.glob("*.yaml")) + list(cards_dir.glob("*.json")))
        ok.append(f"Edge cards directory exists ({card_count} cards)")
    else:
        issues.append(f"Edge cards directory not found: {cards_dir}")

    # Check .env exists
    if Path(".env").exists():
        ok.append(".env file exists")
    else:
        issues.append(".env file not found (run 'opencausality init')")

    # Display results
    for item in ok:
        console.print(f"  [green]OK[/green]  {item}")
    for item in issues:
        console.print(f"  [yellow]!![/yellow]  {item}")

    console.print()
    if not issues:
        console.print("[bold green]All checks passed.[/bold green]")
    else:
        console.print(f"[yellow]{len(issues)} issue(s) found.[/yellow]")


# ============================================================================
# DAG Commands
# ============================================================================

@dag_app.command("run")
def dag_run(
    dag_path: Path = typer.Argument(
        ...,
        help="Path to DAG YAML specification file",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output directory for results",
    ),
    mode: str = typer.Option(
        "EXPLORATION",
        "--mode", "-m",
        help="Run mode: EXPLORATION or CONFIRMATION",
    ),
    max_iterations: int = typer.Option(
        3,
        "--max-iterations",
        help="Maximum iterations in exploration mode",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Verbose output",
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Continue despite DAG validation errors (treat as warnings)",
    ),
):
    """
    Run a DAG specification through the agentic estimation loop.

    Example:
        opencausality dag run config/agentic/dags/example_welfare.yaml

    The command will:
    1. Parse and validate the DAG
    2. Catalog data availability
    3. Select designs for each edge
    4. Run estimations and produce EdgeCards
    5. Evaluate credibility and iterate (in EXPLORATION mode)
    """
    import logging

    if verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    dag_path = resolve_path(dag_path)
    if not dag_path.exists():
        console.print(f"[red]DAG file not found: {dag_path}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]Running DAG: {dag_path}[/bold cyan]\n")

    try:
        from shared.agentic.dag.parser import parse_dag
        from shared.agentic.dag.validator import validate_dag
        from shared.agentic.agent_loop import AgentLoop, AgentLoopConfig

        # Parse DAG
        console.print("[dim]Parsing DAG...[/dim]")
        dag = parse_dag(dag_path)
        console.print(f"  DAG: [cyan]{dag.metadata.name}[/cyan]")
        console.print(f"  Nodes: {len(dag.nodes)}")
        console.print(f"  Edges: {len(dag.edges)}")
        console.print(f"  Target: {dag.metadata.target_node}")

        # Validate DAG
        console.print("\n[dim]Validating DAG...[/dim]")
        validation = validate_dag(dag, raise_on_error=False)

        if not validation.is_valid:
            if force:
                console.print("[bold yellow]DAG validation errors (--force: continuing anyway):[/bold yellow]")
                for issue in validation.errors():
                    console.print(f"  [yellow]WARN[/yellow] {issue.location}: {issue.message}")
            else:
                console.print("[bold red]DAG validation failed![/bold red]")
                for issue in validation.errors():
                    console.print(f"  [red]ERROR[/red] {issue.location}: {issue.message}")
                raise typer.Exit(1)

        if validation.warnings():
            for issue in validation.warnings():
                console.print(f"  [yellow]WARN[/yellow] {issue.location}: {issue.message}")

        console.print("  [green]Validation passed[/green]")

        # Configure loop
        config = AgentLoopConfig(
            mode=mode,
            max_iterations=max_iterations,
            output_dir=output_dir or Path("outputs/agentic"),
            force_run=force,
        )

        # Run loop
        console.print(f"\n[dim]Running agent loop (mode={mode})...[/dim]\n")
        loop = AgentLoop(dag, config)
        report = loop.run()

        # Display results
        console.print("\n" + "=" * 60)
        console.print("[bold green]RUN COMPLETE[/bold green]")
        console.print("=" * 60)

        # Summary table
        table = Table(title="Edge Results")
        table.add_column("Edge", style="cyan")
        table.add_column("Design", style="white")
        table.add_column("Estimate", style="white")
        table.add_column("Rating", style="yellow")
        table.add_column("Score", style="white")

        for summary in report.edge_summaries:
            est = f"{summary.estimate:.4f}" if summary.estimate else "-"
            table.add_row(
                summary.edge_id,
                summary.design,
                est,
                summary.credibility_rating,
                f"{summary.credibility_score:.2f}",
            )

        console.print(table)

        # Blocked edges
        if report.blocked_edges:
            console.print("\n[bold red]Blocked Edges:[/bold red]")
            for edge_id, reason in report.blocked_edges.items():
                console.print(f"  {edge_id}: {reason}")

        # Output location
        console.print(f"\nResults saved to: [cyan]{config.output_dir}[/cyan]")

    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@dag_app.command("validate")
def dag_validate(
    dag_path: Path = typer.Argument(
        ...,
        help="Path to DAG YAML specification file",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
):
    """
    Validate a DAG specification without running estimation.

    Checks:
    - Referential integrity (edge endpoints exist)
    - Temporal cycles
    - Identity dependencies
    - Forbidden controls consistency
    """
    console.print(f"\n[bold cyan]Validating DAG: {dag_path}[/bold cyan]\n")

    try:
        from shared.agentic.dag.parser import parse_dag
        from shared.agentic.dag.validator import validate_dag

        dag = parse_dag(dag_path)
        console.print(f"  DAG: [cyan]{dag.metadata.name}[/cyan]")

        validation = validate_dag(dag, raise_on_error=False)

        if validation.errors():
            console.print("\n[bold red]Errors:[/bold red]")
            for issue in validation.errors():
                console.print(f"  {issue.location}: {issue.message}")

        if validation.warnings():
            console.print("\n[bold yellow]Warnings:[/bold yellow]")
            for issue in validation.warnings():
                console.print(f"  {issue.location}: {issue.message}")

        if validation.forbidden_controls:
            console.print("\n[dim]Forbidden Controls (per edge):[/dim]")
            for edge_id, result in validation.forbidden_controls.items():
                if result.total_forbidden:
                    console.print(f"  {edge_id}: {sorted(result.total_forbidden)}")

        if validation.is_valid:
            console.print("\n[bold green]DAG is valid[/bold green]")
        else:
            console.print("\n[bold red]DAG validation failed[/bold red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1)


@dag_app.command("list")
def dag_list(
    dags_dir: Path = typer.Option(
        Path("config/agentic/dags"),
        "--dir", "-d",
        help="Directory containing DAG YAML files",
    ),
):
    """List available DAG specifications."""
    console.print(f"\n[bold cyan]DAG Specifications in {dags_dir}[/bold cyan]\n")

    if not dags_dir.exists():
        console.print(f"[yellow]Directory does not exist: {dags_dir}[/yellow]")
        return

    dag_files = list(dags_dir.glob("*.yaml")) + list(dags_dir.glob("*.yml"))

    if not dag_files:
        console.print("[dim]No DAG files found[/dim]")
        return

    try:
        from shared.agentic.dag.parser import parse_dag

        table = Table(title="Available DAGs")
        table.add_column("File", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Nodes", style="white")
        table.add_column("Edges", style="white")
        table.add_column("Target", style="yellow")

        for dag_file in sorted(dag_files):
            try:
                dag = parse_dag(dag_file)
                table.add_row(
                    dag_file.name,
                    dag.metadata.name,
                    str(len(dag.nodes)),
                    str(len(dag.edges)),
                    dag.metadata.target_node or "-",
                )
            except Exception as e:
                table.add_row(
                    dag_file.name,
                    f"[red]Error: {e}[/red]",
                    "-", "-", "-",
                )

        console.print(table)

    except ImportError:
        console.print("[yellow]Could not import DAG parser[/yellow]")
        for dag_file in sorted(dag_files):
            console.print(f"  {dag_file.name}")


# ============================================================================
# DAG Viz Command
# ============================================================================

@dag_app.command("viz")
def dag_viz(
    dag_path: Optional[Path] = typer.Argument(
        None,
        help="Path to DAG YAML (default: from settings)",
    ),
    cards_dir: Optional[Path] = typer.Option(
        None, "--cards", "-c",
        help="Path to edge_cards directory",
    ),
    state_file: Optional[Path] = typer.Option(
        None, "--state", "-s",
        help="Path to state.json for open issues",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output HTML file path",
    ),
    llm_annotate: bool = typer.Option(
        False, "--llm-annotate",
        help="Generate LLM annotations for edge tooltips",
    ),
):
    """
    Generate an interactive DAG visualization HTML.

    Example:
        opencausality dag viz config/agentic/dags/kspi_k2_full.yaml
        opencausality dag viz --llm-annotate -o /tmp/viz.html
    """
    from scripts.build_dag_viz import build, DEFAULT_CARDS_DIR, DEFAULT_STATE

    if dag_path is None:
        try:
            from config.settings import get_settings
            dag_path = resolve_path(get_settings().default_dag_path)
        except Exception:
            dag_path = resolve_path("config/agentic/dags/kspi_k2_full.yaml")
    else:
        dag_path = resolve_path(dag_path)

    if not dag_path.exists():
        console.print(f"[red]DAG file not found: {dag_path}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold cyan]Building DAG visualization: {dag_path}[/bold cyan]")
    result = build(
        dag_path=dag_path,
        cards_dir=cards_dir or DEFAULT_CARDS_DIR,
        state_path=state_file or DEFAULT_STATE,
        output_path=output,
        llm_annotate=llm_annotate,
    )
    console.print(f"[green]Written: {result}[/green]")


# ============================================================================
# Agent Commands
# ============================================================================

AGENT_NAMES = ["datascout", "modelsmith", "estimator", "judge"]

@agent_app.command("round")
def agent_round(
    dag: Optional[Path] = typer.Option(None, "--dag", help="Path to DAG YAML"),
    provider: str = typer.Option("codex", "--provider", "-p", help="Provider: codex or claude"),
):
    """Run one full round of all 4 agents sequentially."""
    import subprocess

    script = Path("scripts/agent_loops/run_agent_round.sh")
    if not script.exists():
        console.print(f"[red]Agent round script not found: {script}[/red]")
        raise typer.Exit(1)

    env = os.environ.copy()
    env["PROVIDER"] = provider
    if dag:
        env["DAG_PATH"] = str(dag)

    console.print(f"[bold cyan]Running agent round (provider={provider})[/bold cyan]")
    result = subprocess.run(["bash", str(script)], env=env)
    if result.returncode != 0:
        console.print("[red]Agent round failed[/red]")
        raise typer.Exit(result.returncode)
    console.print("[green]Agent round complete[/green]")


@agent_app.command("run")
def agent_run(
    agent_name: str = typer.Argument(..., help=f"Agent name: {', '.join(AGENT_NAMES)}"),
    provider: str = typer.Option("codex", "--provider", "-p", help="Provider: codex or claude"),
    once: bool = typer.Option(True, "--once/--loop", help="Run once or loop"),
):
    """Run a single agent."""
    import subprocess

    if agent_name not in AGENT_NAMES:
        console.print(f"[red]Unknown agent: {agent_name}. Choose from: {AGENT_NAMES}[/red]")
        raise typer.Exit(1)

    script = Path(f"scripts/agent_loops/{agent_name}_loop.sh")
    if not script.exists():
        console.print(f"[red]Agent script not found: {script}[/red]")
        raise typer.Exit(1)

    env = os.environ.copy()
    env["PROVIDER"] = provider
    env["RUN_ONCE"] = "1" if once else "0"

    console.print(f"[bold cyan]Running {agent_name} (provider={provider}, once={once})[/bold cyan]")
    result = subprocess.run(["bash", str(script)], env=env)
    if result.returncode != 0:
        console.print(f"[red]{agent_name} failed[/red]")
        raise typer.Exit(result.returncode)
    console.print(f"[green]{agent_name} complete[/green]")


@agent_app.command("status")
def agent_status(
    agent_name: Optional[str] = typer.Argument(None, help="Agent name (or all)"),
):
    """Check agent loop status."""
    agents = [agent_name] if agent_name else AGENT_NAMES
    for name in agents:
        pid_file = Path(f"outputs/agentic/{name}/{name}_loop.pid")
        if pid_file.exists():
            pid = pid_file.read_text().strip()
            console.print(f"  [green]{name}[/green]: running (PID {pid})")
        else:
            console.print(f"  [dim]{name}[/dim]: not running")


@agent_app.command("stop")
def agent_stop(
    agent_name: str = typer.Argument(..., help="Agent name to stop"),
):
    """Stop a running agent loop gracefully."""
    stop_file = Path(f"outputs/agentic/{agent_name}/{agent_name}_loop.stop")
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    stop_file.touch()
    console.print(f"[yellow]Stop signal sent to {agent_name}[/yellow]")


@agent_app.command("log")
def agent_log(
    agent_name: str = typer.Argument(..., help="Agent name"),
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines to show"),
):
    """Show recent agent log."""
    import subprocess

    log_path = Path(f"outputs/agentic/logs/{agent_name}_loop.log")
    if not log_path.exists():
        console.print(f"[dim]No log file for {agent_name}[/dim]")
        return
    subprocess.run(["tail", f"-{lines}", str(log_path)])


# ============================================================================
# Loop Commands (Estimation Loop)
# ============================================================================

@loop_app.command("start")
def loop_start(
    dag: Optional[Path] = typer.Option(None, "--dag", help="Path to DAG YAML"),
    provider: str = typer.Option("codex", "--provider", "-p", help="Provider: codex or claude"),
):
    """Start the estimation loop in the background."""
    import subprocess

    script = Path("scripts/codex_loop/control.sh")
    env = os.environ.copy()
    env["PROVIDER"] = provider
    if dag:
        env["DAG_PATH"] = str(dag)

    result = subprocess.run(["bash", str(script), "start"], env=env)
    raise typer.Exit(result.returncode)


@loop_app.command("stop")
def loop_stop():
    """Stop the estimation loop gracefully."""
    import subprocess
    subprocess.run(["bash", "scripts/codex_loop/control.sh", "stop"])


@loop_app.command("status")
def loop_status():
    """Check estimation loop status."""
    import subprocess
    subprocess.run(["bash", "scripts/codex_loop/control.sh", "status"])


@loop_app.command("once")
def loop_once(
    provider: str = typer.Option("codex", "--provider", "-p", help="Provider: codex or claude"),
):
    """Run a single estimation loop iteration."""
    import subprocess

    env = os.environ.copy()
    env["PROVIDER"] = provider
    result = subprocess.run(["bash", "scripts/codex_loop/control.sh", "once"], env=env)
    raise typer.Exit(result.returncode)


@loop_app.command("log")
def loop_log():
    """Show recent estimation loop log."""
    import subprocess
    subprocess.run(["bash", "scripts/codex_loop/control.sh", "log"])


# ============================================================================
# Data Commands
# ============================================================================

@data_app.command("ingest")
def data_ingest(
    force: bool = typer.Option(False, "--force", "-f", help="Re-ingest even if unchanged"),
    file: Optional[Path] = typer.Option(None, "--file", help="Ingest a specific file"),
):
    """Scan data/raw/ and ingest new or changed files."""
    from shared.engine.ingest import IngestPipeline

    pipeline = IngestPipeline()

    if file:
        if not file.exists():
            console.print(f"[red]File not found: {file}[/red]")
            raise typer.Exit(1)
        try:
            profile = pipeline.ingest_file(file, force=force)
            results = [profile]
        except Exception as e:
            console.print(f"[red]Ingest failed: {e}[/red]")
            raise typer.Exit(1)
    else:
        results = pipeline.ingest_all(force=force)

    if not results:
        console.print("[dim]No new or changed files to ingest.[/dim]")
        return

    count = pipeline.register_loaders()

    table = Table(title="Ingested Files")
    table.add_column("File ID", style="cyan")
    table.add_column("Format", style="white")
    table.add_column("Rows", style="white", justify="right")
    table.add_column("Frequency", style="yellow")
    table.add_column("Date Range", style="white")
    table.add_column("Value Columns", style="white")

    for profile in results:
        dr = f"{profile.date_range[0]} to {profile.date_range[1]}" if profile.date_range else "-"
        table.add_row(
            profile.file_id,
            profile.format,
            str(profile.rows),
            profile.frequency or "-",
            dr,
            ", ".join(profile.value_columns[:5]) + ("..." if len(profile.value_columns) > 5 else ""),
        )

    console.print(table)
    console.print(f"\n[green]Ingested {len(results)} file(s), registered {count} loader(s).[/green]")


@data_app.command("list")
def data_list():
    """List all ingested datasets from the manifest."""
    from shared.engine.ingest import IngestPipeline, _slugify_column

    pipeline = IngestPipeline()
    datasets = pipeline.manifest.datasets

    if not datasets:
        console.print("[dim]No ingested datasets. Drop files into data/raw/ and run 'opencausality data ingest'.[/dim]")
        return

    table = Table(title="Ingested Datasets")
    table.add_column("File ID", style="cyan")
    table.add_column("Format", style="white")
    table.add_column("Rows", style="white", justify="right")
    table.add_column("Frequency", style="yellow")
    table.add_column("Date Range", style="white")
    table.add_column("Node IDs", style="dim")
    table.add_column("Ingested At", style="dim")

    for ds in datasets:
        dr = f"{ds.date_range[0]} to {ds.date_range[1]}" if ds.date_range else "-"
        node_ids = [f"{ds.file_id}__{_slugify_column(c)}" for c in ds.value_columns[:3]]
        node_str = ", ".join(node_ids)
        if len(ds.value_columns) > 3:
            node_str += f" (+{len(ds.value_columns) - 3})"
        ingested = ds.ingested_at[:19] if ds.ingested_at else "-"
        table.add_row(
            ds.file_id,
            ds.format,
            str(ds.rows),
            ds.frequency or "-",
            dr,
            node_str,
            ingested,
        )

    console.print(table)


@data_app.command("profile")
def data_profile(
    file_id: str = typer.Argument(..., help="File ID to profile (from 'data list')"),
):
    """Show detailed column-level profile for an ingested dataset."""
    from shared.engine.ingest import IngestPipeline

    pipeline = IngestPipeline()
    ds = pipeline.manifest.get_dataset(file_id)

    if ds is None:
        console.print(f"[red]Dataset not found: {file_id}[/red]")
        available = [d.file_id for d in pipeline.manifest.datasets]
        if available:
            console.print(f"[dim]Available: {', '.join(available)}[/dim]")
        raise typer.Exit(1)

    # Dataset summary panel
    dr = f"{ds.date_range[0]} to {ds.date_range[1]}" if ds.date_range else "N/A"
    summary = (
        f"[bold]File ID:[/bold] {ds.file_id}\n"
        f"[bold]Source:[/bold] {ds.original_path}\n"
        f"[bold]Format:[/bold] {ds.format}\n"
        f"[bold]Rows:[/bold] {ds.rows}\n"
        f"[bold]Date Column:[/bold] {ds.date_column or 'N/A'}\n"
        f"[bold]Frequency:[/bold] {ds.frequency or 'N/A'}\n"
        f"[bold]Date Range:[/bold] {dr}\n"
        f"[bold]Unique Dates:[/bold] {ds.n_unique_dates}\n"
        f"[bold]Duplicate Dates:[/bold] {ds.n_duplicate_dates}\n"
        f"[bold]Output:[/bold] {ds.output_path}\n"
        f"[bold]Ingested:[/bold] {ds.ingested_at}"
    )
    console.print(Panel(summary, title=f"Dataset: {file_id}"))

    # Column details table
    table = Table(title="Column Profiles")
    table.add_column("Column", style="cyan")
    table.add_column("Type", style="yellow")
    table.add_column("Non-null", style="white", justify="right")
    table.add_column("Null", style="white", justify="right")
    table.add_column("Mean", style="white", justify="right")
    table.add_column("Std", style="white", justify="right")
    table.add_column("Min", style="white", justify="right")
    table.add_column("Max", style="white", justify="right")
    table.add_column("Value?", style="green")

    for col in ds.columns:
        is_value = "Y" if col.name in ds.value_columns else ""
        table.add_row(
            col.name,
            col.dtype,
            str(col.non_null_count),
            str(col.null_count),
            f"{col.mean:.4f}" if col.mean is not None else "-",
            f"{col.std:.4f}" if col.std is not None else "-",
            f"{col.min_val:.4f}" if col.min_val is not None else "-",
            f"{col.max_val:.4f}" if col.max_val is not None else "-",
            is_value,
        )

    console.print(table)

    if ds.sidecar:
        console.print(f"\n[dim]Sidecar metadata: {ds.sidecar}[/dim]")


@data_app.command("watch")
def data_watch():
    """Watch data/raw/ for new or modified files and auto-ingest them."""
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        console.print(
            "[red]watchdog not installed. Install with:[/red]\n"
            "  pip install watchdog"
        )
        raise typer.Exit(1)

    import time
    from shared.engine.ingest import IngestPipeline

    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)

    supported = IngestPipeline.SUPPORTED_EXTENSIONS
    managed = IngestPipeline.KNOWN_MANAGED_DIRS

    # Track pending files for stability check
    pending: dict[str, float] = {}  # path -> last_mtime
    STABILITY_SECONDS = 2.0

    class IngestHandler(FileSystemEventHandler):
        def on_created(self, event):
            if not event.is_directory:
                self._enqueue(event.src_path)

        def on_modified(self, event):
            if not event.is_directory:
                self._enqueue(event.src_path)

        def _enqueue(self, src_path: str):
            p = Path(src_path)
            if p.suffix.lower() not in supported:
                return
            if p.suffix.lower() in (".yaml", ".yml"):
                return
            try:
                rel = p.relative_to(raw_dir)
                first_dir = rel.parts[0] if len(rel.parts) > 1 else None
                if first_dir and first_dir.lower() in managed:
                    return
            except ValueError:
                return
            pending[str(p)] = time.time()

    handler = IngestHandler()
    observer = Observer()
    observer.schedule(handler, str(raw_dir), recursive=True)
    observer.start()

    console.print(f"[bold cyan]Watching {raw_dir} for changes...[/bold cyan]")
    console.print("[dim]Press Ctrl+C to stop.[/dim]")

    try:
        while True:
            time.sleep(0.5)
            now = time.time()
            ready = []
            for path_str, enqueue_time in list(pending.items()):
                p = Path(path_str)
                if not p.exists():
                    del pending[path_str]
                    continue
                # Stability check: wait until mtime hasn't changed for STABILITY_SECONDS
                try:
                    current_mtime = p.stat().st_mtime
                except OSError:
                    del pending[path_str]
                    continue
                if now - current_mtime >= STABILITY_SECONDS and now - enqueue_time >= STABILITY_SECONDS:
                    ready.append(path_str)

            for path_str in ready:
                del pending[path_str]
                p = Path(path_str)
                pipeline = IngestPipeline()
                try:
                    profile = pipeline.ingest_file(p, force=True)
                    pipeline.register_loaders()
                    console.print(
                        f"[green]Ingested:[/green] {profile.file_id} "
                        f"({profile.rows} rows, {profile.frequency or '?'} freq, "
                        f"{len(profile.value_columns)} value col(s))"
                    )
                except Exception as e:
                    console.print(f"[red]Failed to ingest {p.name}: {e}[/red]")
    except KeyboardInterrupt:
        observer.stop()
        console.print("\n[dim]Stopped watching.[/dim]")
    observer.join()


# ============================================================================
# Query REPL Command
# ============================================================================

@app.command("monitor")
def monitor(
    output_dir: Path = typer.Option(
        Path("outputs/agentic"),
        "--output-dir", "-o",
        help="Output directory to watch for notifications",
    ),
    auto_open: bool = typer.Option(
        False,
        "--auto-open",
        help="Auto-open HITL panel in browser on notification",
    ),
    poll_interval: float = typer.Option(
        2.0,
        "--interval",
        help="Polling interval in seconds",
    ),
):
    """Watch for HITL notifications and display events in real-time."""
    import json
    import time
    import webbrowser

    sentinel_path = output_dir / ".notification.json"
    last_mtime: float = 0.0

    console.print(f"[bold cyan]Monitoring {output_dir} for notifications...[/bold cyan]")
    console.print("[dim]Press Ctrl+C to stop.[/dim]\n")

    try:
        while True:
            try:
                if sentinel_path.exists():
                    current_mtime = sentinel_path.stat().st_mtime
                    if current_mtime > last_mtime:
                        last_mtime = current_mtime
                        try:
                            with open(sentinel_path) as f:
                                data = json.load(f)
                        except (json.JSONDecodeError, OSError):
                            time.sleep(poll_interval)
                            continue

                        event = data.get("event", "unknown")
                        message = data.get("message", "")
                        ts = data.get("timestamp", "")[:19]
                        run_id = data.get("run_id", "")

                        if event == "hitl_required":
                            pending = data.get("pending_count", 0)
                            panel_path = data.get("panel_path")
                            console.print(
                                f"\n[bold yellow]HITL REQUIRED[/bold yellow] "
                                f"[dim]{ts}[/dim] run={run_id}"
                            )
                            console.print(f"  {message}")
                            if panel_path:
                                console.print(f"  Panel: [cyan]{panel_path}[/cyan]")
                                if auto_open:
                                    try:
                                        webbrowser.open(Path(panel_path).resolve().as_uri())
                                    except Exception:
                                        pass

                        elif event == "run_complete":
                            console.print(
                                f"\n[bold green]RUN COMPLETE[/bold green] "
                                f"[dim]{ts}[/dim] run={run_id}"
                            )
                            console.print(f"  {message}")

                        else:
                            console.print(
                                f"\n[bold]{event}[/bold] [dim]{ts}[/dim]"
                            )
                            console.print(f"  {message}")
                else:
                    pass  # Sentinel doesn't exist yet, keep waiting

            except OSError:
                pass  # File access race condition

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        console.print("\n[dim]Stopped monitoring.[/dim]")


@app.command("query")
def query_repl(
    dag_path: Optional[Path] = typer.Option(None, "--dag", "-d", help="Path to DAG YAML"),
    cards_dir: Optional[Path] = typer.Option(None, "--cards", "-c", help="Edge cards directory"),
    mode: str = typer.Option("REDUCED_FORM", "--mode", "-m", help="Query mode"),
):
    """Start the interactive causal query REPL."""
    from scripts.query_repl import start_repl

    start_repl(dag_path=dag_path, cards_dir=cards_dir, mode=mode)


# ============================================================================
# DAG Generate Command
# ============================================================================

@dag_app.command("generate")
def dag_generate(
    narrative: Path = typer.Argument(
        ...,
        help="Path to narrative text file",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    base_dag: Optional[Path] = typer.Option(
        None, "--base-dag", "-b",
        help="Path to base DAG YAML for node matching",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output path for generated DAG YAML",
    ),
):
    """Generate a DAG from a natural language narrative via LLM extraction."""
    from scripts.generate_narrative_dag import generate, DEFAULT_BASE_DAG, DEFAULT_OUT

    console.print(f"[bold cyan]Generating DAG from narrative: {narrative}[/bold cyan]")
    try:
        result = generate(
            narrative_path=narrative,
            base_dag_path=base_dag or DEFAULT_BASE_DAG,
            output_path=output or DEFAULT_OUT,
        )
        console.print(f"[green]Written: {result}[/green]")
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1)


# ============================================================================
# DAG Compare Command
# ============================================================================

@dag_app.command("compare")
def dag_compare(
    baseline_dir: Path = typer.Argument(
        ...,
        help="Path to baseline (Run A) output directory",
        exists=True,
        file_okay=False,
    ),
    new_dir: Path = typer.Argument(
        ...,
        help="Path to new (Run B) output directory",
        exists=True,
        file_okay=False,
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output path for comparison report",
    ),
):
    """Compare EdgeCards between two estimation runs."""
    from scripts.compare_runs import compare

    console.print(f"[bold cyan]Comparing runs: {baseline_dir} vs {new_dir}[/bold cyan]")
    try:
        result = compare(baseline_dir, new_dir, output)
        console.print(f"[green]Report written: {result}[/green]")
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1)


# ============================================================================
# Data Download Command
# ============================================================================

@data_app.command("download")
def data_download(
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download all sources"),
    verify_only: bool = typer.Option(False, "--verify-only", help="Only verify existing data"),
    source: Optional[str] = typer.Option(None, "--source", "-s", help="Filter by source name"),
):
    """Download all research data sources with status tracking."""
    from scripts.download_all_data import download_data

    console.print("[bold cyan]Data Download[/bold cyan]")
    download_data(force=force, verify_only=verify_only, source_filter=source)


# ============================================================================
# Config Enrich Command
# ============================================================================

@config_app.command("enrich")
def config_enrich(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Don't write files",
    ),
    registry_only: bool = typer.Option(
        False, "--registry-only", help="Only enrich issue_registry",
    ),
    actions_only: bool = typer.Option(
        False, "--actions-only", help="Only enrich hitl_actions",
    ),
):
    """Enrich HITL YAML configs with LLM-generated descriptions."""
    from scripts.enrich_hitl_text import enrich

    console.print("[bold cyan]Enriching HITL text...[/bold cyan]")
    total = enrich(dry_run=dry_run, registry_only=registry_only, actions_only=actions_only)
    console.print(f"[green]Done. Total LLM calls: {total}[/green]")


# ============================================================================
# HITL Commands
# ============================================================================

hitl_app = typer.Typer(
    name="hitl",
    help="Human-in-the-Loop panel commands",
)
app.add_typer(hitl_app, name="hitl")


@hitl_app.command("build")
def hitl_build(
    state_file: Optional[Path] = typer.Option(
        None, "--state-file",
        help="Path to state.json",
    ),
    dag_file: Optional[Path] = typer.Option(
        None, "--dag-file",
        help="Path to DAG YAML for edge/node context",
    ),
    llm_annotate: bool = typer.Option(
        False, "--llm-annotate",
        help="Generate LLM-powered contextual decision guidance",
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o",
        help="Output directory for hitl_panel.html",
    ),
):
    """Build the HITL Resolution Panel HTML."""
    from scripts.build_hitl_panel import (
        build, DEFAULT_STATE, DEFAULT_CARDS_DIR,
        DEFAULT_ACTIONS, DEFAULT_REGISTRY, DEFAULT_OUTPUT_DIR, DEFAULT_DAG,
    )

    console.print("[bold cyan]Building HITL panel...[/bold cyan]")
    try:
        result = build(
            state_path=state_file or DEFAULT_STATE,
            cards_dir=DEFAULT_CARDS_DIR,
            actions_path=DEFAULT_ACTIONS,
            registry_path=DEFAULT_REGISTRY,
            output_dir=output_dir or DEFAULT_OUTPUT_DIR,
            dag_path=dag_file or DEFAULT_DAG,
            llm_annotate=llm_annotate,
        )
        console.print(f"[green]Written: {result}[/green]")
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1)


# =============================================================================
# Benchmark Commands
# =============================================================================

@benchmark_app.command("run")
def benchmark_run(
    suite: str = typer.Option("dgp", help="Benchmark suite: dgp, acic, or all"),
    adapter: Optional[str] = typer.Option(None, help="Specific adapter name to benchmark"),
    format: str = typer.Option("table", help="Output format: table, json, or markdown"),
):
    """Run benchmark suite against adapters."""
    console = Console()
    console.print(f"[bold cyan]Running benchmark suite: {suite}[/bold cyan]")

    summaries = []

    if suite in ("dgp", "all"):
        from benchmarks.dgp_benchmarks import run_all_dgp_benchmarks
        dgp_results = run_all_dgp_benchmarks()
        for name, summary in dgp_results.items():
            if adapter and name != adapter:
                continue
            summaries.append(summary)

    if suite in ("acic", "all"):
        from benchmarks.acic_loader import load_acic_data
        from benchmarks.harness import run_benchmark, summarize_benchmark
        from shared.engine.adapters.registry import get_adapter as get_adapter_fn

        datasets = load_acic_data()
        # ACIC only supports LP (OLS-like) adapter for now
        for adapter_name in ["LOCAL_PROJECTIONS"]:
            if adapter and adapter_name != adapter:
                continue
            try:
                adapter_obj = get_adapter_fn(adapter_name)
                results = run_benchmark(adapter_obj, datasets, target_effect="att")
                summaries.append(summarize_benchmark(results))
            except Exception as e:
                console.print(f"[yellow]Skipping {adapter_name}: {e}[/yellow]")

    if not summaries:
        console.print("[yellow]No benchmark results generated.[/yellow]")
        raise typer.Exit(1)

    if format == "json":
        import json
        console.print(json.dumps([s.to_dict() for s in summaries], indent=2))
    elif format == "markdown":
        import pandas as pd
        df = pd.DataFrame([s.to_dict() for s in summaries])
        console.print(df.to_markdown(index=False))
    else:
        table = Table(title="Benchmark Results")
        table.add_column("Adapter", style="cyan")
        table.add_column("Datasets", justify="right")
        table.add_column("Mean Bias", justify="right")
        table.add_column("RMSE", justify="right")
        table.add_column("Coverage", justify="right")
        table.add_column("Avg Runtime", justify="right")
        for s in summaries:
            table.add_row(
                s.adapter_name,
                str(s.n_datasets),
                f"{s.mean_bias:.4f}",
                f"{s.rmse:.4f}",
                f"{s.coverage_90:.1%}",
                f"{s.mean_runtime:.3f}s",
            )
        console.print(table)


@benchmark_app.command("report")
def benchmark_report(
    format: str = typer.Option("table", help="Output format: table, json, or markdown"),
):
    """Generate a report from the most recent benchmark run."""
    console = Console()
    console.print("[bold cyan]Running full DGP benchmark report...[/bold cyan]")

    from benchmarks.dgp_benchmarks import run_all_dgp_benchmarks

    summaries = run_all_dgp_benchmarks()

    if format == "json":
        import json
        console.print(json.dumps(
            {name: s.to_dict() for name, s in summaries.items()}, indent=2,
        ))
    else:
        table = Table(title="DGP Benchmark Report")
        table.add_column("Adapter", style="cyan")
        table.add_column("Datasets", justify="right")
        table.add_column("Mean Bias", justify="right")
        table.add_column("RMSE", justify="right")
        table.add_column("Coverage", justify="right")
        for name, s in summaries.items():
            table.add_row(
                name,
                str(s.n_datasets),
                f"{s.mean_bias:.4f}",
                f"{s.rmse:.4f}",
                f"{s.coverage_90:.1%}",
            )
        console.print(table)


# ============================================================================
# Discover Commands (Causal Discovery)
# ============================================================================

@discover_app.command("pc")
def discover_pc(
    data: Path = typer.Argument(
        ...,
        help="Path to CSV data file",
        exists=True,
    ),
    alpha: float = typer.Option(0.05, "--alpha", "-a", help="Significance level"),
    dag: Optional[Path] = typer.Option(None, "--dag", help="Existing DAG YAML to compare against"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON file"),
):
    """Run PC algorithm for constraint-based causal discovery."""
    _run_discovery("PC", data, dag, output, alpha=alpha)


@discover_app.command("ges")
def discover_ges(
    data: Path = typer.Argument(
        ...,
        help="Path to CSV data file",
        exists=True,
    ),
    dag: Optional[Path] = typer.Option(None, "--dag", help="Existing DAG YAML to compare against"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON file"),
):
    """Run GES (Greedy Equivalence Search) algorithm."""
    _run_discovery("GES", data, dag, output)


@discover_app.command("notears")
def discover_notears(
    data: Path = typer.Argument(
        ...,
        help="Path to CSV data file",
        exists=True,
    ),
    lambda1: float = typer.Option(0.1, "--lambda", "-l", help="L1 penalty for sparsity"),
    dag: Optional[Path] = typer.Option(None, "--dag", help="Existing DAG YAML to compare against"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON file"),
):
    """Run NOTEARS algorithm for continuous structure learning."""
    _run_discovery("NOTEARS", data, dag, output, lambda1=lambda1)


@discover_app.command("compare")
def discover_compare(
    dag: Path = typer.Argument(
        ...,
        help="Path to DAG YAML",
        exists=True,
    ),
    result: Path = typer.Argument(
        ...,
        help="Path to discovery result JSON",
        exists=True,
    ),
):
    """Compare a discovery result against an existing DAG."""
    import json

    from shared.agentic.dag.parser import parse_dag
    from shared.agentic.agents.discovery_agent import DiscoveryAgent, DiscoveryResult, DiscoveredEdge

    existing_dag = parse_dag(dag)

    with open(result) as f:
        data = json.load(f)

    disc_result = DiscoveryResult(
        algorithm=data.get("algorithm", "unknown"),
        edges_discovered=[
            DiscoveredEdge(
                from_node=e["from_node"],
                to_node=e["to_node"],
                edge_type=e.get("edge_type", "directed"),
                weight=e.get("weight", 1.0),
            )
            for e in data.get("edges", [])
        ],
        node_names=data.get("node_names", []),
    )

    agent = DiscoveryAgent()
    comparison = agent.compare_with_dag(disc_result, existing_dag)

    table = Table(title="Discovery vs DAG Comparison")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Edges", style="white")

    for category in ["confirmed", "contradicted", "novel", "missing"]:
        edges = comparison[category]
        edge_strs = [f"{e.from_node} -> {e.to_node}" for e in edges[:5]]
        suffix = f" (+{len(edges) - 5})" if len(edges) > 5 else ""
        table.add_row(
            category.capitalize(),
            str(len(edges)),
            ", ".join(edge_strs) + suffix if edge_strs else "-",
        )

    console.print(table)


def _run_discovery(
    algorithm: str,
    data_path: Path,
    dag_path: Optional[Path],
    output_path: Optional[Path],
    **kwargs,
):
    """Internal helper to run a discovery algorithm."""
    import json

    import pandas as pd

    from shared.agentic.agents.discovery_agent import DiscoveryAgent

    console.print(f"[bold cyan]Running {algorithm} discovery on {data_path}[/bold cyan]")

    df = pd.read_csv(data_path)
    agent = DiscoveryAgent()

    if algorithm == "PC":
        result = agent.run_pc(df, alpha=kwargs.get("alpha", 0.05))
    elif algorithm == "GES":
        result = agent.run_ges(df)
    elif algorithm == "NOTEARS":
        result = agent.run_notears(df, lambda1=kwargs.get("lambda1", 0.1))
    else:
        console.print(f"[red]Unknown algorithm: {algorithm}[/red]")
        raise typer.Exit(1)

    # Display results
    console.print(f"  Algorithm: {result.algorithm}")
    console.print(f"  Edges discovered: {len(result.edges_discovered)}")
    console.print(f"  Nodes: {len(result.node_names)}")

    table = Table(title="Discovered Edges")
    table.add_column("From", style="cyan")
    table.add_column("To", style="cyan")
    table.add_column("Type", style="yellow")
    table.add_column("Weight", justify="right")

    for edge in result.edges_discovered:
        table.add_row(
            edge.from_node,
            edge.to_node,
            edge.edge_type,
            f"{edge.weight:.3f}",
        )

    console.print(table)

    # Compare with existing DAG if provided
    if dag_path:
        from shared.agentic.dag.parser import parse_dag

        existing_dag = parse_dag(dag_path)
        comparison = agent.compare_with_dag(result, existing_dag)

        console.print("\n[bold]Comparison with existing DAG:[/bold]")
        for category in ["confirmed", "contradicted", "novel", "missing"]:
            edges = comparison[category]
            color = {"confirmed": "green", "contradicted": "red", "novel": "yellow", "missing": "dim"}[category]
            console.print(f"  [{color}]{category.capitalize()}: {len(edges)}[/{color}]")

    # Save output
    if output_path:
        output_data = {
            "algorithm": result.algorithm,
            "node_names": result.node_names,
            "edges": [
                {
                    "from_node": e.from_node,
                    "to_node": e.to_node,
                    "edge_type": e.edge_type,
                    "weight": e.weight,
                }
                for e in result.edges_discovered
            ],
            "metadata": result.metadata,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        console.print(f"\n[green]Saved to {output_path}[/green]")


if __name__ == "__main__":
    app()
