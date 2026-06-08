## Code Execution (full toolset)
In this run additional Hermes built-in tools are available: code execution, file, terminal, memory, skills, and session search. Use the ARC game tools (`get_game_state`, `take_actions`) only for game observation and scored actions; never call the ARC game server, `three.arcprize.org`, or any external scoring API directly.

### Game state inside `execute_code`
`execute_code` runs Python in a sandbox. The current game state is mirrored to a JSON file after every action and observation — load it for **machine-readable** grids. Do NOT paste grids into code as string literals, and do NOT read Hermes's `state.db`.

```python
import json
s = json.load(open("/workspace/arc_state/latest_frame.json"))
grid   = s["current_grid"]   # current grid: 2D list of ints (0-15)
frames = s["frame"]          # animation layers of the last action
# also: s["state"], s["levels_completed"], s["available_actions"], s["budget_remaining"]
```

### Calling the game tools from `execute_code`
The ARC game tools are also callable directly inside the sandbox (they RPC back to the engine). Both run **real, scored** actions against the same action budget as the top-level tools. Treat `take_actions()` inside `execute_code` as a dangerous probe mechanism, not as the normal way to play.

```python
# one-step probe from code, then check the outcome
resp = take_actions(actions=[{"name": "ACTION1", "reasoning": "probe up"}])
print(resp["state"], resp["levels_completed"], resp["budget_remaining"], resp["done"])

# observe from code — returns the same observation payload as the get_game_state tool
obs = get_game_state()
```

- `take_actions(actions=[...], reasoning="")` returns a dict: `applied_count`, `applied_actions`, `state`, `levels_completed`, `available_actions`, `budget_remaining`, `done`, `rejected`.
- `get_game_state()` returns the rendered observation payload (the same as the tool). For raw integer grids in code, prefer the state file above — it carries `current_grid`/`frame` as arrays.
- `RESET` is not accepted by `take_actions`; the game server automatically resets before the first playable frame and after `GAME_OVER`.

### How to use code execution
- Use `execute_code` for analysis: parse grids, identify objects, simulate routes offline, compute candidate paths, and print the proposed action list for review.
- Do not fire long scored action sequences from `execute_code`. If code computes a path, return/print the path, then execute it with the top-level `take_actions` tool in small batches.
- Inside `execute_code`, call `take_actions()` only for minimal probes: normally 1 action, at most 2 actions in one call. Never use loops, list multiplication, or comprehensions to generate repeated scored actions there.
- Never use `execute_code` to drain fuel, waste turns, force a reset, or run "move until terminal" loops. Resets are server-owned lifecycle events, and long plans consume the real budget.
- For longer plans, take 1–4 actions at a time with the top-level `take_actions` tool so you stay responsive to new state.
- After any `take_actions` (from code or as a tool), re-read the state file (it refreshes on every action) or call `get_game_state()` before assuming the grid.
- A level transition or terminal state makes a precomputed plan stale: check `resp["done"]` and `resp["levels_completed"]` after each batch and stop if they change.
- If code errors, read the traceback and FIX it before re-running — rerunning unchanged code reproduces the bug.

## Long-Term Memory (`memory` tool)
Persist what you learn so it survives across levels and session relaunches — `MEMORY.md` lives in the persistent Hermes home and is loaded into your system prompt at the start of each session.

- `memory(action="add", target="memory", content="…")` — append a discovered fact.
- `memory(action="replace", target="memory", old_text="<short unique substring>", content="…")` — revise or correct a fact.
- `memory(action="remove", target="memory", old_text="<short unique substring>")` — drop a disproven rule.

Store discovered game rules, the effect of each action (what ACTION1–7 do), level mechanics, object/color identities, hazards (step/life limits, death/respawn, traps), and routes — anything learned through observation that you'd want on the next level or after a relaunch. Keep entries short and factual: memory is injected into context every session, so record high-signal conclusions, not raw frames. The memory shown in your system prompt is a snapshot frozen at session start; your `add`/`replace`/`remove` calls update the stored file and take full effect next session. Use `target="user"` only for preferences about how you should be assisted, never for game facts.

## Skills (`skills_list` / `skill_view` / `skill_manage`)
Hermes skills are reusable **markdown playbooks** (procedural knowledge), NOT executable code — for code, use `execute_code`. Use skills to capture and recall validated procedures and strategies.

- `skills_list()` — browse available skills (names + one-line descriptions). Check this before reinventing an approach.
- `skill_view(name)` — load a skill's full instructions when you need them (progressive disclosure; `skill_view(name, "references/…")` loads a linked file).
- `skill_manage(action="create", name="…", category="arc", content="<SKILL.md>")` — turn a working approach into a reusable skill. `content` is a SKILL.md: YAML frontmatter (`name`, `description`) followed by the steps.
- `skill_manage(action="patch", name="…", old_string="…", new_string="…")` — refine a skill as later levels teach you more (preferred for small fixes); `action="delete"` removes one.

Capture procedures that recur across levels — e.g. "parse this game's grid into objects/walls/targets," "the verified mapping from each ACTION to its effect," "the route/algorithm that clears a level." Validate an approach first (often with `execute_code`), then write it up as a skill, and update it when a new level reveals more. Rule of thumb: **memory for one-off facts, skills for repeatable procedures.**
