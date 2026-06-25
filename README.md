# Continual Harness on ARC-AGI-3

**Continual Harness** is a reset-free, self-improving agentic harness for long-horizon
interactive tasks. It enables test-time learning for foundation models within a
single run. 

This repo applies Continual Harness to **ARC-AGI-3**, an interactive reasoning
benchmark whose games must be played without provided rules or domain-specific knowledge.

- 📝 Blog post: [Continual Harness: An Efficient Self-Improving Agent on ARC-AGI-3](BLOG_URL_PLACEHOLDER)
- 📄 Paper: [Continual Harness: Online Adaptation for Self-Improving Foundation Agents](https://arxiv.org/abs/2605.09998)

> The repo is cloned from [arcprize/ARC-AGI-3-Agents](https://github.com/arcprize/ARC-AGI-3-Agents).
> We reuse the upstream scaffolding, including base `Agent`, `Swarm`, `Playback`. The Continual
> Harness agent and the Hermes evaluation harness are the original additions.

## Repository layout

| Path | What it is |
| --- | --- |
| `main.py` | Entry point for Continual Harness. The original swarm launcher from the upstream repo: it builds a `Swarm` that runs one agent instance per game, each on its own thread. |
| `run_cli.py` | Entry point for the **Hermes** baseline harness — runs the external Hermes agent in a Docker container against ARC games over an MCP proxy. |
| `agents/templates/continual_harness_agent.py` | The `ContinualHarness` agent: the orchestrator loop that drives the engine and the four editable stores. |
| `agents/templates/continual_harness/` | Continual Harness internals — memory, skills, subagents, prompts, the sandbox, and the Refiner (`harness_evolver.py`). |
| `agents/` | Base agent / swarm scaffolding and per-run artifact handling. |
| `cli_agents/` | Hermes harness infrastructure: ARC game server, FastMCP proxy, and agent backends. |
| `logs/` | Per-run artifacts: isolated memory / skills / subagents / prompt snapshots, traces, trajectories, and recordings. |

## Requirements

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- An **ARC-AGI-3 API key** (`ARC_API_KEY`) — get one from the
  [ARC-AGI-3 website](https://three.arcprize.org/).
- A **Gemini API key** (`GEMINI_API_KEY`).
- **Docker** — only needed for the Hermes baseline.

## Installation

1. Clone the repo and enter the directory.

```bash
git clone https://github.com/feng-rrRay/Continual-Harness-ARC-AGI-3.git
cd Continual-Harness-ARC-AGI-3
```

2. Copy the example environment file.

```bash
cp .env.example .env
```

3. Set your API keys in `.env`:

```bash
ARC_API_KEY="your_arc_api_key_here"
GEMINI_API_KEY="your_gemini_api_key_here"
```

`uv` resolves and installs the dependencies automatically on the first `uv run`, so
there is no separate install step.

## Running Continual Harness

Run Continual Harness on a single game (for example, `ft09`):

```bash
uv run main.py --agent=continualharness --game=ft09
```

- `--game` accepts comma-separated prefixes (e.g. `--game=ft09,ka59`). Omit `--game` to
  run a swarm over every game the API returns, one thread per game.

Per-run artifacts are written to `logs/continualharness-<game>-<timestamp>/`, with one
isolated subfolder per game.

## Running Hermes baseline

Hermes evaluation uses the `hermes-eval` optional dependencies and runs Hermes in a
Docker container built from `.devcontainer/hermes-agent/`. Make sure Docker is available
and that `ARC_API_KEY` and `GEMINI_API_KEY` are set.

Build the Hermes container image on the first run:

```bash
uv run --extra hermes-eval python run_cli.py --game ls20 --build
```

After the image is built, launch Hermes without rebuilding:

```bash
uv run --extra hermes-eval python run_cli.py --game ls20
```

By default Hermes receives only the ARC MCP tools (`get_game_state` and `take_actions`).
To expose the full Hermes built-in toolset, pass `--toolset full`. Logs are written under
`logs/hermes-<timestamp>/<game>/`; gameplay recordings are exposed directly under that
run's `recordings/` directory.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for
details.
