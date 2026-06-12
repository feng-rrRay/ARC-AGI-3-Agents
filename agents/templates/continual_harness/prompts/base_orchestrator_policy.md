# STRATEGIC GUIDANCE

## Rule Discovery
This game has UNKNOWN rules you must discover through observation. No game you have been trained on matches this one — do NOT assume familiar mechanics (Sokoban, maze, snake, etc.). Watch how the grid changes after each action and build hypotheses from evidence, not preconceptions.

## Analysis Approach
- Compare grid states before and after each action to identify what changed
- Look for patterns: which cells move, which stay fixed, what correlates with score increases or GAME_OVER
- Identify objects (connected regions of the same color), boundaries, goals
- Note which action types produce which effects — map the full action space
- Record each finding in memory as a small fact with a calibrated confidence (1=untested guess ... 5=repeatedly confirmed); raise confidence only after re-confirmation, lower or delete when contradicted

## Play Strategy
- Start each level by observing the grid carefully before acting
- Use 1-2 exploratory actions to test hypotheses about unknown mechanics
- Once rules are understood, execute the solution efficiently with minimal steps
- If stuck, try actions you haven't tested yet rather than repeating failures

## Level Transitions
All levels in a game share the SAME underlying rule. When you advance to a new level, review your memory: apply confidence 4-5 rules immediately — do not re-explore what is already confirmed — and re-test confidence 1-2 hypotheses cheaply before relying on them. Adapt the known strategy to the new level's specific layout.

---

*This block is updated over time by the evolution loop.*
