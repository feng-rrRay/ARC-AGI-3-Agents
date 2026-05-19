## CURRENT SYSTEM INSTRUCTION
{current_prompt}

## GAME STATE
state={state} score={score} action_counter={action_counter} generation={generation}

## RECENT TRAJECTORY (last {n} steps)
{trajectory}

## LONG-TERM MEMORY
{memory_overview}

## SKILLS
{skill_overview}

## SUBAGENTS
{subagent_overview}

## CURRENT FRAME
{frame}

# TASK
Propose an improved system instruction. Call evolve_system_prompt(reasoning, new_prompt).
