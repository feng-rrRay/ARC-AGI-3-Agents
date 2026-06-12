## Main Agent's System Prompt (FIXED - Cannot Be Changed):
The main agent receives this as its system message every step. You must NOT duplicate it into the base prompt. It lists tools and hard constraints. Use it so you know what tools the agent has when improving strategic guidance.

{system_prompt}

---

## Current Base Prompt (Strategic Guidance - YOUR TARGET):
This is the optimizable strategic guidance that gets combined with runtime context (action history, game state, tool results) and sent to the main agent. You can modify this to improve the agent's decision-making.

{current_base_prompt}

## Recent Agent Trajectories (last {n} steps):
{trajectory}

## Evolution Trigger
{trigger_context}

## Current Agent State
{memory_overview}

{skill_overview}

{subagent_overview}

## Your Task
Analyze the agent's recent performance and create an IMPROVED base prompt. Treat the base prompt as the orchestrator's durable playbook for this game: it should preserve discovered rules AND improve the agent's general acting policy for progressing through future levels.

The improved prompt should include:

1. **Confidence-calibrated game rules** — Preserve confidence **4-5** memory entries as confirmed rules. Carry confidence **3** entries as working assumptions, explicitly labeled as unverified. Put confidence **1-2** entries under "Open questions / to verify" and never state them as facts. Losing high-confidence rules forces re-exploration and wastes actions; hardening low-confidence guesses into rules is just as costly.
2. **State representation policy** — Describe how the orchestrator should interpret each new grid: object identities, color roles, connected components, coordinates, boundaries, goals, hazards, movable entities, changed cells, and action-effect mappings. Prefer compact operational representations over vague visual descriptions.
3. **Action-selection policy** — Explain the best current procedure for choosing actions: what to inspect first, how to derive candidate moves, when to test hypotheses, when to execute a planned sequence, and how to use recent history to avoid repeated failures.
4. **Tool usage policy** — Include durable guidance on when to use memory, skills, subagents, or code-like analysis. For example: use memory to preserve stable rules; use skills/code-style reasoning for coordinate-heavy planning or repeated transformations; use subagents only for separable analysis; avoid tool calls that duplicate information already visible in the prompt.
5. **Failure and stagnation policy** — If the trajectory shows loops, no-ops, invalid actions, wrong assumptions, or GAME_OVER causes, add concrete recovery guidance. The policy should tell the orchestrator how to revise hypotheses and try informative alternatives instead of repeating the same action pattern.
6. **Targeted improvement only** — Keep guidance that is working. Do not duplicate the fixed system prompt or tool schemas. Do not include raw trajectory dumps. Keep the prompt compact, operational, and directly useful for the next decision.

## Analysis Guidelines
Look for these patterns:
- Repeated failures (stuck in loops, wrong tool usage, bumping walls, repeating no-ops)
- Successful strategies (efficient play, correct rule application, good memory/skill usage)
- Progress toward objectives (score increases, level transitions)
- Rule discovery (has the agent figured out mechanics but not recorded them in the prompt?)
- Miscalibrated confidence (does the trajectory contradict a confidence-5 entry, or repeatedly confirm a confidence-1 one? Flag it in the prompt so the agent re-checks and edits the entry)
- Not adapting when stuck → emphasize flexibility and trying new approaches

## Output Format
Provide the complete improved base prompt as markdown below. Make targeted improvements, keeping the elements that are working while adding guidance where needed.

IMPROVED BASE PROMPT:
