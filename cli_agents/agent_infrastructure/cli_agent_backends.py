"""CLI agent backend abstraction for the ARC-AGI-3 Hermes evaluation.

Mirrors utils/agent_infrastructure/cli_agent_backends.py from pokeagent-speedrun,
adapted for ARC. Only HermesCliBackend is implemented.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session metrics (thin container; extended in phase-2 parity pass)
# ---------------------------------------------------------------------------

@dataclass
class CliSessionMetrics:
    total_cost_usd: float = 0.0
    total_turns: int = 0
    auth_fatal_error: bool = False
    is_error: bool = False
    last_error: str = ""


@dataclass
class CliSession:
    process: subprocess.Popen
    stop_event: threading.Event
    stream_thread: threading.Thread
    metrics: CliSessionMetrics = field(default_factory=CliSessionMetrics)


# ---------------------------------------------------------------------------
# HermesCliBackend
# ---------------------------------------------------------------------------

class HermesCliBackend:
    """Builds and monitors the Hermes Docker container for one ARC game."""

    # Paths inside the container
    AGENT_MEMORY_PATH = "/home/hermes-agent/.hermes"
    WORKSPACE_PATH = "/workspace"
    PROJECT_ROOT_PATH = "/opt/arc-src"

    # Docker image name (built from .devcontainer/hermes-agent/)
    container_image = "arc-hermes-agent"
    devcontainer_build_context = ".devcontainer/hermes-agent"

    # Config file markers — same begin/end convention as pokeagent
    CONFIG_MARKER_BEGIN = "# BEGIN ARC HERMES MCP"
    CONFIG_MARKER_END   = "# END ARC HERMES MCP"

    # ---------------------------------------------------------------------------
    # Docker image build
    # ---------------------------------------------------------------------------

    def build_image(
        self,
        project_root: str | Path,
        *,
        hermes_commit: str = "",
        user_uid: int | None = None,
        user_gid: int | None = None,
    ) -> None:
        """Build the Docker image from .devcontainer/hermes-agent/Dockerfile."""
        root = Path(project_root)
        uid = user_uid or os.getuid()
        gid = user_gid or os.getgid()
        build_args: list[str] = [
            "--build-arg", f"USER_UID={uid}",
            "--build-arg", f"USER_GID={gid}",
        ]
        if hermes_commit:
            build_args += ["--build-arg", f"HERMES_COMMIT={hermes_commit}"]
        cmd = [
            "docker", "build",
            "-t", self.container_image,
            *build_args,
            "-f", str(root / self.devcontainer_build_context / "Dockerfile"),
            str(root / self.devcontainer_build_context),
        ]
        logger.info("Building Docker image: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
        logger.info("Docker image %s built successfully", self.container_image)

    # ---------------------------------------------------------------------------
    # MCP config injection
    # ---------------------------------------------------------------------------

    def _build_mcp_config_block(self, mcp_port: int) -> str:
        return (
            f"{self.CONFIG_MARKER_BEGIN}\n"
            "mcp_servers:\n"
            "  arc-agi-3:\n"
            f'    url: "http://host.docker.internal:{mcp_port}/mcp"\n'
            "    tools:\n"
            "      prompts: false\n"
            "      resources: false\n"
            f"{self.CONFIG_MARKER_END}\n"
        )

    def inject_mcp_config(self, hermes_memory_dir: Path, mcp_port: int) -> None:
        """Write or update the ARC MCP block in Hermes config.yaml."""
        hermes_memory_dir.mkdir(parents=True, exist_ok=True)
        config_path = hermes_memory_dir / "config.yaml"
        block = self._build_mcp_config_block(mcp_port)

        if config_path.exists():
            text = config_path.read_text()
            # Remove any existing ARC MCP block
            text = re.sub(
                rf"{re.escape(self.CONFIG_MARKER_BEGIN)}.*?{re.escape(self.CONFIG_MARKER_END)}\n?",
                "",
                text,
                flags=re.DOTALL,
            )
        else:
            text = ""
        config_path.write_text(text.rstrip("\n") + "\n" + block)
        logger.debug("Injected MCP config into %s (port=%d)", config_path, mcp_port)

    # ---------------------------------------------------------------------------
    # Docker run command
    # ---------------------------------------------------------------------------

    def build_launch_cmd(
        self,
        directive_path: str | Path,
        game_id: str,
        hermes_memory_dir: Path,
        scratch_dir: Path,
        project_root: Path,
        run_id: str,
        mcp_port: int,
        game_port: int,
        model: str = "gemini-3.1-pro-preview",
        provider: str = "gemini",
        api_key_env: str = "GEMINI_API_KEY",
        max_turns: int = 90,
        resume_session_id: str = "",
    ) -> list[str]:
        """Return the full `docker run` command list."""
        wrapper_cmd = [
            "python3",
            f"{self.PROJECT_ROOT_PATH}/cli_agents/agent_infrastructure/hermes_wrapper.py",
            "--directive-path", f"{self.WORKSPACE_PATH}/.agent_directive.txt",
            "--working-dir",    self.WORKSPACE_PATH,
            "--server-url",     f"http://host.docker.internal:{game_port}",
            "--hermes-home",    self.AGENT_MEMORY_PATH,
            "--model",          model,
            "--provider",       provider,
            "--api-key-env",    api_key_env,
            "--max-turns",      str(max_turns),
        ]
        if resume_session_id:
            wrapper_cmd += ["--resume-session-id", resume_session_id]

        docker_cmd = [
            "docker", "run", "--rm",
            "--name", f"arc-hermes-{run_id}-{game_id}",
            "--cap-add=NET_ADMIN",
            "--security-opt=seccomp=unconfined",
            "--network=bridge",
            "--add-host=host.docker.internal:host-gateway",
            "-v", f"{hermes_memory_dir}:{self.AGENT_MEMORY_PATH}",
            "-v", f"{scratch_dir}:{self.WORKSPACE_PATH}",
            "-v", f"{project_root}:{self.PROJECT_ROOT_PATH}:ro",
            "-w", self.WORKSPACE_PATH,
            "-e", f"MCP_PORT={mcp_port}",
            "-e", f"GAME_SERVER_PORT={game_port}",
            "-e", f"RUN_DATA_ID={run_id}",
            "-e", f"HERMES_HOME={self.AGENT_MEMORY_PATH}",
            "-e", f"PYTHONPATH={self.PROJECT_ROOT_PATH}",
            "-e", f"HERMES_MODEL={model}",
            "-e", f"HERMES_PROVIDER={provider}",
            "-e", f"HERMES_API_KEY_ENV={api_key_env}",
            "-e", f"HERMES_MAX_TURNS={max_turns}",
        ]

        # Pass through Gemini / Google API keys (never ARC keys)
        passthrough = [
            "GEMINI_API_KEY", "GOOGLE_API_KEY",
            "HERMES_BASE_URL", "HERMES_DISABLE_MULTIMODAL",
            "HERMES_API_TIMEOUT", "HERMES_VISION_TIMEOUT",
        ]
        for var in passthrough:
            val = os.environ.get(var)
            if val:
                docker_cmd += ["-e", f"{var}={val}"]

        docker_cmd.append(self.container_image)
        docker_cmd.extend(wrapper_cmd)
        return docker_cmd

    # ---------------------------------------------------------------------------
    # Stream reader — phase-1: tee to log; phase-2 adds JSONL event parsing
    # ---------------------------------------------------------------------------

    def run_stream_reader(
        self,
        stdout_pipe: io.RawIOBase,
        stop_event: threading.Event,
        log_file: io.TextIOWrapper | None,
        metrics: CliSessionMetrics | None,
        server_url: str | None = None,
    ) -> None:
        """Read container stdout; tee to log_file; parse JSONL events."""
        try:
            buffered = io.BufferedReader(stdout_pipe)  # type: ignore[arg-type]
            for raw_line in buffered:
                if stop_event.is_set():
                    break
                line = raw_line.decode("utf-8", errors="replace")
                if log_file:
                    log_file.write(line)
                    log_file.flush()
                stripped = line.strip()
                if not stripped:
                    continue
                # Phase-1: best-effort JSONL parse for the result event only
                try:
                    event = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                self._handle_stream_event(event, metrics)
        except Exception as exc:
            logger.debug("stream reader exited: %s", exc)

    def _handle_stream_event(
        self, event: dict[str, Any], metrics: CliSessionMetrics | None
    ) -> None:
        etype = event.get("type", "")
        if etype == "result" and metrics:
            metrics.total_cost_usd += float(event.get("total_cost_usd") or 0)
            metrics.total_turns    += int(event.get("num_turns") or 0)
            metrics.is_error        = bool(event.get("is_error"))
            metrics.last_error      = str(event.get("error") or "")
        elif etype == "error" and metrics:
            msg = str(event.get("message") or "")
            if "auth" in msg.lower() or "unauthorized" in msg.lower():
                metrics.auth_fatal_error = True
            metrics.last_error = msg
        # Phase-2: thinking / tool_use events forwarded to the game server
