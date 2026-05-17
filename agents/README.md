# ARC-AGI-3 Agents

For all information on how to build, test, and run agents, as well as the technical specifications of the agent and game APIs, please see the [agents documentation](https://three.arcprize.org/docs#agent-quickstart).

## Rendering Recordings To GIF

Use the local recording renderer to convert `.recording.jsonl` files into GIFs:

```bash
uv run python -m agents.recording_render \
  recordings/ls20-9607627b.continualharness.gemini-3.1-pro-preview.f4d2cccb-3757-4f75-925b-68206f13dd8c.recording.jsonl \
  --output recordings/ls20-9607627b.continualharness.gif \
  --fps 5 \
  --scale 8
```

The renderer reads frame events from the JSONL file, renders ARC grid values
`0..15` as pixel art, and expands multi-grid frame events into sequential GIF
frames. Metadata overlays include step, game, state, score progress, the chosen
action, and available actions.

Useful options:

```bash
--output PATH              # Output GIF path; defaults beside the recording
--fps N                    # Playback speed
--scale N                  # Pixel-art cell scale
--no-overlay               # Render only the game frame
--actions-log logs/run.log # Use a specific run log for action labels
```

For older recordings where `action_input` was not preserved correctly, the
renderer auto-discovers a matching `logs/*.log` file when possible and uses it
to display the correct action labels instead of showing every action as
`RESET`.
