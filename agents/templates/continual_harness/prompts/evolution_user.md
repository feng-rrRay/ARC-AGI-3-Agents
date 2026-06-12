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
Analyze the agent's recent performance and create an IMPROVED base prompt that:
1. **Addresses observed failures** — if the agent made mistakes, add specific guidance to prevent them
2. **Reinforces successful patterns** — if certain strategies worked well, emphasize them
3. **Includes ALL discovered game rules** — you MUST preserve every confirmed or hypothesized rule the agent has found. Losing rules forces re-exploration and wastes actions.
4. **Adds learned lessons** — include insights derived from trajectory analysis

## Analysis Guidelines
Look for these patterns:
- Repeated failures (stuck in loops, wrong tool usage, bumping walls, repeating no-ops)
- Successful strategies (efficient play, correct rule application, good memory/skill usage)
- Progress toward objectives (score increases, level transitions)
- Rule discovery (has the agent figured out mechanics but not recorded them in the prompt?)
- Not adapting when stuck → emphasize flexibility and trying new approaches

## Output Format
Provide the complete improved base prompt as markdown below. Make targeted improvements, keeping the elements that are working while adding guidance where needed.

IMPROVED BASE PROMPT:
