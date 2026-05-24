from __future__ import annotations

# NOTE: deliberately minimal top-level imports. Anything imported here is also
# imported in the worker process (sandbox.py runs as a script via --worker), and
# every module imported in the worker becomes reachable via class-MRO traversal
# (().__class__.__mro__[-1].__subclasses__()). Heavy parent-only modules like
# `subprocess`, `tempfile`, `logging`, `pathlib`, `select`, `time` are imported
# INSIDE run_python_snippet() so they never load in the worker.
import json
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

# --- Caps (parent-side; child also enforces via rlimit when available) -------

CODE_MAX_CHARS = 8000
ARGS_JSON_MAX_CHARS = 4000
STDOUT_CAP = 5000
STDERR_CAP = 5000
RESULT_CAP = 5000
DEFAULT_TIMEOUT_S = 30.0  # raised from 5s — skills now drive engine via RPC
MAX_IMAGES = 16              # cap on images ferried into the sandbox per call
IMAGE_BYTES_CAP = 256 * 1024  # per-image encoded PNG byte cap

# Type for the RPC callback supplied by the parent. Receives the RPC method
# name and args dict; must return a JSON-serialisable response dict shaped
# like {"ok": bool, "value": ..., "error": "..."}.
RpcCallback = Callable[[str, dict[str, Any]], dict[str, Any]]

# Canonical ARC 16-colour palette. Duplicated from `helpers.py:18-35` so the
# worker subprocess can render grids without importing parent-side modules
# (the worker is locked down with -I/-S and uses a small set of modules from
# `_safe_modules`). Skill code calls `render_grid` / `render_grids` from the
# globals namespace to convert raw grid data to PIL.Image objects using this
# palette — output is pixel-identical to `helpers.frame_to_images`.
_ARC_PALETTE: list[tuple[int, int, int, int]] = [
    (0xFF, 0xFF, 0xFF, 0xFF),
    (0xCC, 0xCC, 0xCC, 0xFF),
    (0x99, 0x99, 0x99, 0xFF),
    (0x66, 0x66, 0x66, 0xFF),
    (0x33, 0x33, 0x33, 0xFF),
    (0x00, 0x00, 0x00, 0xFF),
    (0xE5, 0x3A, 0xA3, 0xFF),
    (0xFF, 0x7B, 0xCC, 0xFF),
    (0xF9, 0x3C, 0x31, 0xFF),
    (0x1E, 0x93, 0xFF, 0xFF),
    (0x88, 0xD8, 0xF1, 0xFF),
    (0xFF, 0xDC, 0x00, 0xFF),
    (0xFF, 0x85, 0x1B, 0xFF),
    (0x92, 0x12, 0x31, 0xFF),
    (0x4F, 0xCC, 0x30, 0xFF),
    (0xA3, 0x56, 0xD6, 0xFF),
]


# --- Worker-side resource limits (Unix best-effort) --------------------------
# Adjusted from the pre-image-support defaults to accommodate numpy + Pillow:
#   AS: 256 MB → 2 GB (numpy alone needs ~150-200 MB at import).
#   CPU: 5 s → 30 s (matches DEFAULT_TIMEOUT_S; lets numpy do real work).
#   NOFILE: 32 → 256 (numpy/PIL open multiple shared libs and codec files).
#   NPROC: 1 → 4096. NPROC counts user-wide threads on Linux (not just this
#     process); OpenBLAS pre-allocates threads at numpy import time and the
#     ceiling has to clear existing user processes too. We don't allow
#     subprocess.fork via the audit hook, so this isn't a fork-bomb vector.

RLIMIT_CPU_SECONDS = 30
RLIMIT_AS_BYTES = 2 * 1024 * 1024 * 1024
RLIMIT_FSIZE_BYTES = 1 * 1024 * 1024
RLIMIT_NOFILE = 256
RLIMIT_NPROC = 4096

_BANNED_NAMES = {
    "__builtins__",
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}

# `getattr` / `hasattr` are allowed by name and as attribute references — skill
# code uses them for ordinary read-only introspection. Dynamic dunder access
# (`getattr(obj, "__class__")`) is still blocked by the dunder-string check in
# `_validate_code`, and dangerous attributes (`f_globals`, `tb_frame`, …) are
# still listed below.
_BANNED_ATTRIBUTES = {
    "__builtins__",
    "__import__",
    "breakpoint",
    "delattr",
    "dir",
    "eval",
    "exec",
    "f_back",
    "f_builtins",
    "f_globals",
    "f_locals",
    "func_globals",
    "gi_frame",
    "globals",
    "help",
    "input",
    "locals",
    "mro",
    "open",
    "setattr",
    "tb_frame",
    "vars",
}

# Events that should NEVER fire from user code, regardless of what stdlib does
# internally. Note: `open`, `compile`, `exec`, and `import` are intentionally
# NOT here — those fire as side effects of legitimate stdlib operations (e.g.
# Counter.most_common lazy-imports heapq, which opens its .pyc file). User-side
# `open()`/`compile()`/`exec()`/`import` are already blocked by the AST validator
# (names in _BANNED_NAMES, import statements rejected outright).
_BANNED_AUDIT_EVENTS = {
    "os.chdir",
    "os.chmod",
    "os.chown",
    "os.exec",
    "os.fork",
    "os.forkpty",
    "os.kill",
    "os.link",
    "os.listdir",
    "os.mkdir",
    "os.open",
    "os.posix_spawn",
    "os.remove",
    "os.rename",
    "os.rmdir",
    "os.scandir",
    "os.spawn",
    "os.startfile",
    "os.symlink",
    "os.system",
    "os.truncate",
    "os.unlink",
    "pathlib.Path.glob",
    "pathlib.Path.rglob",
    "shutil.copyfile",
    "shutil.copymode",
    "shutil.copystat",
    "shutil.copytree",
    "shutil.move",
    "shutil.rmtree",
    "socket.__new__",
    "socket.bind",
    "socket.connect",
    "socket.getaddrinfo",
    "socket.gethostbyaddr",
    "socket.gethostbyname",
    "socket.gethostname",
    "socket.sendmsg",
    "socket.sendto",
    "subprocess.Popen",
}

_BANNED_AUDIT_PREFIXES = (
    "ctypes.",
    "ftplib.",
    "glob.",
    "http.client.",
    "imaplib.",
    "nntplib.",
    "poplib.",
    "posix.",
    "smtplib.",
    "socket.",
    "sqlite3.",
    "ssl.",
    "subprocess.",
    "telnetlib.",
    "urllib.",
)

# Module names blocked at `import` audit time. The AST validator only accepts
# imports of names in `_PRELOADED_MODULE_NAMES`; this list is the runtime
# backstop for lazy/internal imports a banned module might still trigger.
# stdlib lazy imports like `heapq` (used by Counter.most_common) are
# intentionally NOT here.
_BANNED_IMPORT_MODULES = frozenset(
    {
        "ctypes",
        "ftplib",
        "http",
        "http.client",
        "imaplib",
        "nntplib",
        "os",
        "pathlib",
        "poplib",
        "posix",
        "shutil",
        "smtplib",
        "socket",
        "sqlite3",
        "ssl",
        "subprocess",
        "telnetlib",
        "urllib",
        "urllib.request",
    }
)

# Modules that `_safe_modules()` pre-binds into the worker's globals dict. The
# AST validator allows `import X` / `from X import Y` only when every root in
# the statement is in this set. The pre-bound binding still wins at runtime
# (the import becomes a no-op in practice for `import X`, while `from X import
# Y` succeeds against the real module — both forms are intentional). Keep this
# in sync with `_safe_modules()` below.
_PRELOADED_MODULE_NAMES: frozenset[str] = frozenset(
    {
        "Image",
        "ImageChops",
        "ImageDraw",
        "ImageFilter",
        "ImageOps",
        # `PIL` is the real package root; `from PIL import Image` is the
        # natural idiom even though `Image` is also pre-bound directly.
        "PIL",
        "collections",
        "copy",
        "dataclasses",
        "functools",
        "hashlib",
        "heapq",
        "itertools",
        "json",
        "math",
        "np",
        "numpy",
        "random",
        "re",
        "statistics",
    }
)


@dataclass(frozen=True, slots=True)
class SandboxState:
    """JSON-only read-only view passed to sandboxed code as `state`."""

    latest_frame: dict[str, Any] | None = None
    recent_trajectory: list[dict[str, Any]] = field(default_factory=list)
    memory_entries: list[dict[str, Any]] = field(default_factory=list)
    skill_entries: list[dict[str, Any]] = field(default_factory=list)


# ======================== PARENT ============================================


def run_python_snippet(
    code: str,
    *,
    state: SandboxState,
    args: dict[str, Any] | None = None,
    images: list[Any] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    on_rpc: RpcCallback | None = None,
) -> dict[str, Any]:
    """Run sandboxed Python with optional bidirectional RPC.

    Spawns a child Python process, sends a JSON payload, and then enters a
    read loop on the child's stdout (line-delimited JSON). The child may
    emit:
      * `{"type": "rpc", "method": ..., "args": {...}}` — request to the
        parent. We dispatch via `on_rpc(method, args)` and write the response
        back on the child's stdin as a line of JSON.
      * `{"type": "result", "success": ..., "result": ..., "stdout": ..., "stderr": ...}`
        — final result; the read loop exits.

    If `on_rpc` is None, any RPC from the worker is rejected and the worker
    is expected to handle the error (or the user code will raise).

    `images` may be a list of PIL.Image objects (typically the output of
    `frame_to_images(latest_frame)`); they are PNG+base64-encoded into the
    payload and decoded inside the worker as `state.images`. Capped at
    MAX_IMAGES and IMAGE_BYTES_CAP per image — entries that overflow either
    cap are silently dropped (skill code reading `state.images` must handle
    a shorter-than-expected list).

    Never raises. Returns a dict with `success` always present. On timeout
    or spawn failure, returns a synthetic error dict.
    """
    # Parent-only imports stay local so the worker doesn't pull subprocess /
    # tempfile / pathlib / select / time / base64 / io into its class-MRO
    # surface.
    import base64
    import io
    import select
    import subprocess
    import tempfile
    import time
    from pathlib import Path

    if not isinstance(code, str) or not code.strip():
        return {"success": False, "error": "code is empty"}
    if len(code) > CODE_MAX_CHARS:
        return {"success": False, "error": f"code exceeds {CODE_MAX_CHARS} chars"}
    validation_error = _validate_code(code)
    if validation_error is not None:
        return {"success": False, "error": validation_error}

    args = dict(args or {})
    try:
        args_json = json.dumps(args)
    except (TypeError, ValueError) as exc:
        return {"success": False, "error": f"args not JSON-serializable: {exc}"}
    if len(args_json) > ARGS_JSON_MAX_CHARS:
        return {"success": False, "error": f"args exceed {ARGS_JSON_MAX_CHARS} chars"}

    encoded_images: list[str] = []
    if images:
        for img in list(images)[:MAX_IMAGES]:
            try:
                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=False)
                data = buf.getvalue()
            except Exception:
                continue  # malformed PIL object — drop it, don't fail the call
            if len(data) > IMAGE_BYTES_CAP:
                continue
            encoded_images.append(base64.b64encode(data).decode("ascii"))

    # The worker is launched with `-I -S` (no PYTHONPATH, no site.py) for
    # safety, but it still needs to import numpy + PIL for skill code to
    # work. We compute the site-packages directories holding those two
    # packages and ferry them in the payload; the worker inserts them into
    # sys.path early in `_worker_main`. This is surgical — only the
    # specified directories become discoverable, not the full user/system
    # site-packages cascade that site.py would have wired up.
    sandbox_extra_paths = _collect_runtime_paths()

    # `default=str` is a defensive fallback: the agent already converts the
    # frame via model_dump(mode="json"), but if any trajectory or memory entry
    # happens to contain a non-JSON value (rare), we stringify rather than
    # crash the sandbox call.
    try:
        payload = json.dumps(
            {
                "code": code,
                "args": args,
                "state": asdict(state),
                "images": encoded_images,
                "rpc_enabled": on_rpc is not None,
                "sandbox_extra_paths": sandbox_extra_paths,
            },
            default=str,
        )
    except (TypeError, ValueError) as exc:
        return {"success": False, "error": f"sandbox payload not JSON-serializable: {exc}"}

    worker_script = str(Path(__file__).resolve())
    # -I: ignore PYTHON* env vars and don't add cwd to sys.path.
    # -S: skip site.py and site-packages.
    # -u: unbuffered stdout/stderr (critical for line-by-line RPC).
    cmd = [sys.executable, "-I", "-S", "-u", worker_script, "--worker"]
    env = {
        "PYTHONNOUSERSITE": "1",
        "LANG": "C",
        "PATH": "",
        # Cap BLAS / OpenMP thread pools. On many-core machines (e.g. 192
        # cores), numpy's default of one-thread-per-core will allocate ~1.5 GB
        # of thread stacks at import — colliding with the worker's RLIMIT_AS.
        # Skills do small array ops; a single thread is fine.
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }

    deadline = time.monotonic() + max(0.1, float(timeout_s))
    proc: subprocess.Popen | None = None
    try:
        cwd_ctx = tempfile.TemporaryDirectory(prefix="ch-sandbox-")
        cwd = cwd_ctx.__enter__()
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # line-buffered
                cwd=cwd,
                env=env,
                start_new_session=True,
            )
        except OSError as exc:
            cwd_ctx.__exit__(None, None, None)
            return {"success": False, "error": f"sandbox spawn failed: {exc}"}

        try:
            # Send initial payload + newline so worker's readline returns.
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(payload + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            _kill_proc(proc)
            return {"success": False, "error": f"sandbox stdin write failed: {exc}"}

        result_obj: dict[str, Any] | None = None
        protocol_error: str | None = None

        # Main read loop: read line-delimited JSON from worker's stdout,
        # dispatch RPCs back via on_rpc, accept the final {"type":"result"}.
        try:
            while True:
                line = _readline_until(proc.stdout, deadline)
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    msg = json.loads(stripped)
                except json.JSONDecodeError:
                    # Worker emitted non-JSON on the protocol stream (likely
                    # user `print()` that leaked past our redirect, or a crash
                    # traceback). Skip the line and keep reading; if the
                    # worker dies, the EOF branch will fire below.
                    continue
                if not isinstance(msg, dict):
                    continue
                mtype = msg.get("type")
                if mtype == "rpc":
                    method = str(msg.get("method") or "")
                    rpc_args = msg.get("args") or {}
                    if not isinstance(rpc_args, dict):
                        resp: dict[str, Any] = {"ok": False, "error": "rpc args must be an object"}
                    elif on_rpc is None:
                        resp = {"ok": False, "error": "rpc disabled by parent"}
                    else:
                        try:
                            resp = on_rpc(method, rpc_args)
                        except Exception as exc:
                            resp = {"ok": False, "error": f"rpc handler raised: {exc!r}"}
                    try:
                        proc.stdin.write(json.dumps(resp, default=str) + "\n")
                        proc.stdin.flush()
                    except (BrokenPipeError, OSError) as exc:
                        protocol_error = f"sandbox stdin write failed during rpc: {exc}"
                        break
                elif mtype == "result":
                    result_obj = msg
                    break
                # Unknown message types are ignored; the worker keeps running.
        except TimeoutError:
            stderr_text = _drain_stderr_then_kill(proc)
            cwd_ctx.__exit__(None, None, None)
            return {
                "success": False,
                "error": f"sandbox timeout after {timeout_s:.1f}s",
                "stdout": "",
                "stderr": stderr_text[:STDERR_CAP],
            }
        except EOFError:
            stderr_text = _drain_stderr_then_kill(proc)
            cwd_ctx.__exit__(None, None, None)
            return {
                "success": False,
                "error": "sandbox worker exited without final result",
                "stdout": "",
                "stderr": stderr_text[:STDERR_CAP],
            }
        finally:
            stderr_final = ""
            if result_obj is not None or protocol_error is not None:
                # Normal exit path: worker either gave us a result or had a
                # protocol error mid-conversation. Drain stderr (if still open)
                # and clean up.
                try:
                    if proc.stderr is not None and not proc.stderr.closed:
                        stderr_final = proc.stderr.read()
                except Exception:
                    stderr_final = ""
                _kill_proc(proc)
                cwd_ctx.__exit__(None, None, None)

        if protocol_error is not None:
            return {"success": False, "error": protocol_error, "stdout": "", "stderr": stderr_final[:STDERR_CAP]}

        if result_obj is None:
            return {
                "success": False,
                "error": "sandbox produced no result",
                "stdout": "",
                "stderr": stderr_final[:STDERR_CAP],
            }

        # Apply hard caps to text fields.
        result_obj.setdefault("success", False)
        result_obj["stdout"] = str(result_obj.get("stdout") or "")[:STDOUT_CAP]
        result_obj["stderr"] = str(result_obj.get("stderr") or stderr_final)[:STDERR_CAP]
        if "result" in result_obj:
            try:
                rendered = json.dumps(result_obj["result"], default=str)
            except (TypeError, ValueError):
                rendered = repr(result_obj["result"])
            if len(rendered) > RESULT_CAP:
                result_obj["result"] = rendered[: RESULT_CAP - 1] + "…"
                result_obj["result_truncated"] = True
            else:
                try:
                    result_obj["result"] = json.loads(rendered)
                except json.JSONDecodeError:
                    result_obj["result"] = rendered
        # Drop our protocol-only `type` field so callers see a clean dict.
        result_obj.pop("type", None)
        return result_obj
    except Exception as exc:
        if proc is not None:
            _kill_proc(proc)
        return {"success": False, "error": f"sandbox host error: {exc!r}"}


def _collect_runtime_paths() -> list[str]:
    """Return site-packages directories the worker must add to sys.path.

    Discovers the directories holding numpy and Pillow at import time in the
    parent (these are already loaded as part of the agent). Both packages
    typically live in the same venv site-packages, so the result usually
    contains a single path. Returns [] if either package fails to import
    (in which case skill code that uses them will get a NameError, but the
    rest of the sandbox still works).
    """
    from pathlib import Path

    paths: list[str] = []
    seen: set[str] = set()
    for pkg_name in ("numpy", "PIL"):
        try:
            module = __import__(pkg_name)
        except ImportError:
            continue
        file_attr = getattr(module, "__file__", None)
        if not file_attr:
            continue
        site_packages = str(Path(file_attr).resolve().parent.parent)
        if site_packages not in seen:
            paths.append(site_packages)
            seen.add(site_packages)
    return paths


def _readline_until(pipe, deadline) -> str:
    """Blocking readline with a wall-clock deadline.

    Raises TimeoutError on deadline, EOFError on pipe close.
    """
    import select
    import time

    fd = pipe.fileno()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        ready, _, _ = select.select([fd], [], [], remaining)
        if not ready:
            raise TimeoutError
        line = pipe.readline()
        if line == "":
            raise EOFError
        return line


def _drain_stderr_then_kill(proc) -> str:
    """Read whatever stderr is currently available (non-blocking), then kill."""
    import select

    out = ""
    try:
        if proc.stderr is not None and not proc.stderr.closed:
            # Drain whatever is ready without blocking — the worker may still
            # be alive, so a blocking read could hang here.
            while True:
                ready, _, _ = select.select([proc.stderr.fileno()], [], [], 0.05)
                if not ready:
                    break
                chunk = proc.stderr.readline()
                if not chunk:
                    break
                out += chunk
                if len(out) >= STDERR_CAP:
                    break
    except Exception:
        pass
    _kill_proc(proc)
    return out


def _kill_proc(proc) -> None:
    """Best-effort kill; swallow errors so callers always proceed."""
    try:
        if proc.poll() is None:
            proc.kill()
    except Exception:
        pass
    try:
        proc.wait(timeout=2.0)
    except Exception:
        pass
    for stream in (proc.stdin, proc.stdout, proc.stderr):
        try:
            if stream is not None:
                stream.close()
        except Exception:
            pass


def _validate_code(code: str) -> str | None:
    """Reject syntax that commonly escapes Python-level sandboxing.

    This is deliberately conservative. The sandbox is for short analysis
    snippets, so losing import statements and dunder/reflection access is an
    acceptable tradeoff for blocking accidental filesystem/network escapes.
    """
    import ast

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return f"SyntaxError: {exc.msg}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            # `import x, y, z` is allowed only when every root is pre-loaded.
            for alias in node.names:
                root = (alias.name or "").split(".", 1)[0]
                if root not in _PRELOADED_MODULE_NAMES:
                    return (
                        f"sandbox policy: import of {alias.name!r} is not "
                        "allowed; pre-loaded modules are already in scope"
                    )
            continue
        if isinstance(node, ast.ImportFrom):
            # `from x import a, b` — allowed only when the source module is
            # pre-loaded. Relative imports (level > 0) are rejected outright.
            if node.level and node.level > 0:
                return "sandbox policy: relative imports are not allowed"
            module = node.module or ""
            root = module.split(".", 1)[0]
            if not module or root not in _PRELOADED_MODULE_NAMES:
                return (
                    f"sandbox policy: import from {module!r} is not allowed; "
                    "pre-loaded modules are already in scope"
                )
            continue
        if isinstance(node, ast.Name):
            if "__" in node.id or node.id in _BANNED_NAMES:
                return f"sandbox policy: name {node.id!r} is not allowed"
        elif isinstance(node, ast.Attribute):
            if (
                node.attr.startswith("_")
                or "__" in node.attr
                or node.attr in _BANNED_ATTRIBUTES
            ):
                return f"sandbox policy: attribute {node.attr!r} is not allowed"
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "__" in node.value:
                return "sandbox policy: dunder strings are not allowed"
    return None


# ======================== WORKER (executes in the child) =====================


_SAFE_BUILTIN_NAMES: tuple[str, ...] = (
    "abs",
    "all",
    "any",
    "ascii",
    "bin",
    "bool",
    "bytearray",
    "bytes",
    "callable",
    "chr",
    "complex",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "getattr",
    "hasattr",
    "hash",
    "hex",
    "id",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "object",
    "oct",
    "ord",
    "pow",
    "print",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "type",
    "zip",
    # Exceptions:
    "Exception",
    "ValueError",
    "KeyError",
    "IndexError",
    "TypeError",
    "ZeroDivisionError",
    "ArithmeticError",
    "RuntimeError",
    "AssertionError",
    "StopIteration",
    "AttributeError",
    "LookupError",
    "NameError",
    "OverflowError",
    "True",
    "False",
    "None",
)


def _safe_builtins() -> dict[str, Any]:
    """Return a small dict of builtins the user code may reference.

    `__import__` is included even though user code can't reference it directly
    (the AST validator blocks every name containing `__`). It's needed because
    numpy 2.x and other modern libraries perform lazy submodule imports at
    runtime — e.g. the first call to `np.array(...)` may trigger a
    `__import__("numpy._core.numeric")` under the hood, which fails with
    KeyError if the symbol isn't in `__builtins__`.
    """
    import builtins as _b

    out = {
        name: getattr(_b, name) for name in _SAFE_BUILTIN_NAMES if hasattr(_b, name)
    }
    out["__import__"] = _b.__import__
    out["__build_class__"] = _b.__build_class__   # required for `class` defs in user code
    return out


def _install_audit_hook(*, allow_exec_events: int = 0) -> None:
    """Block filesystem, network, subprocess, and late imports in the worker."""

    remaining_allowed_exec = allow_exec_events

    def _guard(event: str, args: tuple[Any, ...]) -> None:
        nonlocal remaining_allowed_exec
        if event == "exec" and remaining_allowed_exec > 0:
            remaining_allowed_exec -= 1
            return
        if event == "import":
            # Allow internal stdlib lazy imports (e.g. heapq pulled in by
            # Counter.most_common); block only known-dangerous modules. The AST
            # validator already rejects `import` statements in user code, so
            # imports here come from stdlib internals or compiled code paths.
            module_name = args[0] if args else ""
            if isinstance(module_name, str):
                root = module_name.split(".", 1)[0]
                if (
                    module_name in _BANNED_IMPORT_MODULES
                    or root in _BANNED_IMPORT_MODULES
                ):
                    raise RuntimeError(
                        f"sandbox policy blocked import of {module_name!r}"
                    )
            return
        if event in _BANNED_AUDIT_EVENTS or event.startswith(_BANNED_AUDIT_PREFIXES):
            raise RuntimeError(f"sandbox policy blocked audit event: {event}")

    try:
        sys.addaudithook(_guard)
    except Exception:
        # Audit hooks are best-effort; parent/worker validation and the parent
        # timeout still apply.
        pass


# Numpy top-level attributes intentionally removed before binding. These all
# perform filesystem or network I/O at attacker-controlled paths; the audit
# hook would catch most of them at runtime, but it's cleaner to deny them
# at the namespace level so skill code gets an AttributeError early.
# Submodules (`np.lib`, `np.core`, `np.testing`, `np.distutils`, `np.f2py`)
# are filtered separately by `_public_namespace`'s ModuleType check.
_NUMPY_DENYLIST: frozenset[str] = frozenset(
    {
        "load", "save", "savez", "savez_compressed",
        "loadtxt", "savetxt", "genfromtxt", "recfromtxt", "recfromcsv",
        "fromfile", "tofile", "memmap",
        "DataSource", "compare_chararrays",
    }
)


def _safe_modules() -> dict[str, Any]:
    import collections
    import copy
    import dataclasses
    import functools
    import hashlib
    import heapq
    import itertools
    import json as _json
    import math
    import random
    import re as _re
    import statistics
    import types as _types

    # Image-processing libraries (Pillow, numpy). Imported here so they live
    # in the worker process. Submodules are pulled in once and stay reachable
    # via the namespaces we build below.
    import numpy as _np
    from PIL import (
        Image as _PILImage,
        ImageChops as _ImageChops,
        ImageDraw as _ImageDraw,
        ImageFilter as _ImageFilter,
        ImageOps as _ImageOps,
    )

    def _module_namespace(module: Any, names: tuple[str, ...]) -> Any:
        return _types.SimpleNamespace(
            **{name: getattr(module, name) for name in names if hasattr(module, name)}
        )

    def _public_namespace(module: Any) -> Any:
        attrs: dict[str, Any] = {}
        for name in dir(module):
            if name.startswith("_"):
                continue
            value = getattr(module, name)
            if isinstance(value, _types.ModuleType):
                continue
            attrs[name] = value
        return _types.SimpleNamespace(**attrs)

    def _safe_numpy_namespace(np_mod: Any) -> Any:
        ns = _public_namespace(np_mod)
        for name in _NUMPY_DENYLIST:
            if hasattr(ns, name):
                delattr(ns, name)
        return ns

    numpy_ns = _safe_numpy_namespace(_np)

    return {
        "collections": _module_namespace(
            collections,
            (
                "ChainMap",
                "Counter",
                "OrderedDict",
                "defaultdict",
                "deque",
                "namedtuple",
            ),
        ),
        "copy": _module_namespace(copy, ("copy", "deepcopy")),
        "dataclasses": _module_namespace(
            dataclasses,
            (
                "FrozenInstanceError",
                "InitVar",
                "KW_ONLY",
                "MISSING",
                "asdict",
                "astuple",
                "dataclass",
                "field",
                "fields",
                "is_dataclass",
                "replace",
            ),
        ),
        "functools": _public_namespace(functools),
        "hashlib": _module_namespace(
            hashlib,
            (
                "blake2b",
                "blake2s",
                "md5",
                "new",
                "sha1",
                "sha224",
                "sha256",
                "sha384",
                "sha512",
            ),
        ),
        "heapq": _module_namespace(
            heapq,
            (
                "heapify",
                "heappush",
                "heappop",
                "heappushpop",
                "heapreplace",
                "merge",
                "nlargest",
                "nsmallest",
            ),
        ),
        "itertools": _public_namespace(itertools),
        "json": _module_namespace(
            _json,
            ("JSONDecodeError", "JSONDecoder", "JSONEncoder", "dumps", "loads"),
        ),
        "math": _public_namespace(math),
        "random": _module_namespace(
            random,
            (
                "Random",
                "choice",
                "choices",
                "randint",
                "random",
                "randrange",
                "sample",
                "seed",
                "shuffle",
                "uniform",
            ),
        ),
        "re": _module_namespace(
            _re,
            (
                "A",
                "ASCII",
                "DOTALL",
                "I",
                "IGNORECASE",
                "M",
                "MULTILINE",
                "Match",
                "Pattern",
                "S",
                "VERBOSE",
                "X",
                "compile",
                "escape",
                "findall",
                "finditer",
                "fullmatch",
                "match",
                "search",
                "split",
                "sub",
            ),
        ),
        "statistics": _public_namespace(statistics),
        # --- Image processing libraries ---
        # Pillow: curated whitelists; `Image.open` and `Image.frombuffer` are
        # intentionally NOT exposed because both accept arbitrary file inputs.
        # Skills construct images via `Image.new`, `Image.fromarray`, or our
        # `render_grid` / `render_grids` helpers (added to globals_dict).
        "Image": _module_namespace(
            _PILImage,
            (
                "new",
                "fromarray",
                "merge",
                "blend",
                "alpha_composite",
                "composite",
                "Image",  # the class itself (for isinstance checks)
                "NEAREST",
                "BILINEAR",
                "BICUBIC",
                "LANCZOS",
                "FLIP_LEFT_RIGHT",
                "FLIP_TOP_BOTTOM",
                "ROTATE_90",
                "ROTATE_180",
                "ROTATE_270",
                "TRANSPOSE",
                "TRANSVERSE",
            ),
        ),
        "ImageDraw": _module_namespace(_ImageDraw, ("Draw",)),
        "ImageFilter": _public_namespace(_ImageFilter),
        "ImageOps": _module_namespace(
            _ImageOps,
            (
                "autocontrast",
                "colorize",
                "crop",
                "deform",
                "equalize",
                "expand",
                "fit",
                "flip",
                "grayscale",
                "invert",
                "mirror",
                "pad",
                "posterize",
                "scale",
                "solarize",
            ),
        ),
        "ImageChops": _public_namespace(_ImageChops),
        # numpy: full public surface MINUS file-I/O functions and submodules.
        # Both `np` and `numpy` aliases are provided to match common idioms.
        "np": numpy_ns,
        "numpy": numpy_ns,
    }


def _render_grid_impl(grid: Any) -> Any:
    """Render a 2D integer grid (palette indices) into a PIL.Image (RGBA).

    Exposed to skill code as `render_grid(...)`. Output is pixel-identical
    to `helpers.grid_to_image` (same palette, same byte layout). Useful for
    rendering the post-action frame returned from a
    `tools.take_actions` RPC, where the skill receives raw grid data
    rather than pre-rendered images.
    """
    from PIL import Image as _PILImage

    palette = _ARC_PALETTE
    try:
        height = len(grid)
    except TypeError:
        return _PILImage.new("RGBA", (64, 64), palette[0])
    width = max((len(row) for row in grid), default=0)
    if height == 0 or width == 0:
        return _PILImage.new("RGBA", (64, 64), palette[0])
    raw = bytearray()
    for row in grid:
        for x in range(width):
            try:
                value = row[x]
            except (IndexError, TypeError):
                value = 0
            try:
                raw.extend(palette[int(value) % len(palette)])
            except (TypeError, ValueError):
                raw.extend(palette[0])
    return _PILImage.frombytes("RGBA", (width, height), bytes(raw))


def _render_grids_impl(grids: Any) -> Any:
    """Render a 3D grid stack (a FrameData.frame value) into PIL.Images.

    Exposed to skill code as `render_grids(...)`. Returns a list with one
    image per grid layer. An empty / None input yields a single 64×64
    blank image so callers don't have to special-case zero-length frames.
    """
    from PIL import Image as _PILImage

    if not grids:
        return [_PILImage.new("RGBA", (64, 64), _ARC_PALETTE[0])]
    return [_render_grid_impl(g) for g in grids]


def _apply_rlimits() -> None:
    """Best-effort. Each setrlimit is independently try/except."""
    try:
        import resource
    except ImportError:
        return
    for name, soft, hard in (
        ("RLIMIT_CPU", RLIMIT_CPU_SECONDS, RLIMIT_CPU_SECONDS),
        ("RLIMIT_AS", RLIMIT_AS_BYTES, RLIMIT_AS_BYTES),
        ("RLIMIT_FSIZE", RLIMIT_FSIZE_BYTES, RLIMIT_FSIZE_BYTES),
        ("RLIMIT_NOFILE", RLIMIT_NOFILE, RLIMIT_NOFILE),
        ("RLIMIT_NPROC", RLIMIT_NPROC, RLIMIT_NPROC),
    ):
        lim = getattr(resource, name, None)
        if lim is None:
            continue
        try:
            resource.setrlimit(lim, (soft, hard))
        except (ValueError, OSError):
            pass


class _DualAccess(dict):
    """JSON-style dict that also supports attribute access on string keys.

    Lets skill code read the same payload either way: `state.latest_frame`
    and `state["latest_frame"]` both work, and the same applies recursively
    to nested dicts. Models routinely confuse the two access styles; making
    them equivalent removes a large class of run_skill failures.
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def _wrap(value: Any, _depth: int = 0) -> Any:
    """Recursively wrap dicts as _DualAccess; pass lists element-wise.

    Cap recursion at 8 levels to defend against pathological self-referential
    payloads (everything we ferry through is JSON-serialized, so legitimate
    nesting is shallow).
    """
    if _depth >= 8:
        return value
    if isinstance(value, dict):
        return _DualAccess(
            {k: _wrap(v, _depth + 1) for k, v in value.items()}
        )
    if isinstance(value, list):
        return [_wrap(v, _depth + 1) for v in value]
    return value


def _worker_main() -> int:
    import io
    from contextlib import redirect_stderr, redirect_stdout
    from types import SimpleNamespace

    # Capture the REAL stdin/stdout BEFORE entering redirect contexts so RPC
    # I/O can bypass the user-code redirect buffers.
    real_stdin = sys.stdin
    real_stdout = sys.stdout

    raw = real_stdin.readline()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        real_stdout.write(json.dumps({"type": "result", "success": False, "error": f"bad input: {exc}"}) + "\n")
        real_stdout.flush()
        return 0

    _apply_rlimits()

    code = str(payload.get("code") or "")
    args = dict(payload.get("args") or {})
    state_dict = dict(payload.get("state") or {})
    rpc_enabled = bool(payload.get("rpc_enabled"))
    images_field = payload.get("images") or []

    # Surgical sys.path extension: parent told us where numpy + PIL live. We
    # prepend those directories so subsequent imports inside this worker can
    # find them. The worker still runs with `-I -S`, so no other site-packages
    # cascade leaks in.
    extra_paths = payload.get("sandbox_extra_paths") or []
    if isinstance(extra_paths, list):
        for p in extra_paths:
            if isinstance(p, str) and p and p not in sys.path:
                sys.path.insert(0, p)

    # Decode base64-PNG images into PIL.Image objects BEFORE installing the
    # audit hook. Decoding goes through PIL.Image.open(BytesIO), which can
    # touch lazy-imported codec modules; doing it here keeps the audit hook
    # from intercepting those legitimate stdlib imports. The decoded images
    # are exposed to skill code as `state.images`.
    decoded_images: list[Any] = []
    if isinstance(images_field, list) and images_field:
        import base64 as _b64
        import io as _io_mod

        from PIL import Image as _PILImage

        for s in images_field:
            if not isinstance(s, str):
                continue
            try:
                buf = _io_mod.BytesIO(_b64.b64decode(s))
                decoded_images.append(_PILImage.open(buf).copy())
            except Exception:
                continue
    state_dict["images"] = decoded_images

    # `_wrap` makes both `state.foo` and `state["foo"]` legal; `state.images`
    # is a list of PIL.Image objects and passes through unchanged.
    state = _wrap(state_dict)
    validation_error = _validate_code(code)
    if validation_error is not None:
        real_stdout.write(json.dumps({"type": "result", "success": False, "error": validation_error}) + "\n")
        real_stdout.flush()
        return 0

    try:
        compiled_code = compile(code, "<continual-harness-sandbox>", "exec")
    except SyntaxError as exc:
        real_stdout.write(
            json.dumps({"type": "result", "success": False, "error": f"SyntaxError: {exc.msg}"}) + "\n"
        )
        real_stdout.flush()
        return 0

    safe_modules = _safe_modules()
    _install_audit_hook(allow_exec_events=1)

    # Counts RPCs invoked by the skill so the post-exec no-output warning
    # can distinguish "skill did literally nothing" from "skill drove the
    # engine but skipped `result`".
    rpc_invocations = [0]

    def _make_rpc_tool(method: str):
        def call(**kwargs):
            rpc_invocations[0] += 1
            msg = json.dumps({"type": "rpc", "method": method, "args": kwargs}, default=str) + "\n"
            real_stdout.write(msg)
            real_stdout.flush()
            resp_line = real_stdin.readline()
            if not resp_line:
                raise RuntimeError("sandbox: parent closed pipe")
            try:
                resp = json.loads(resp_line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"sandbox: invalid rpc response: {exc}") from None
            if not isinstance(resp, dict):
                raise RuntimeError("sandbox: rpc response is not a dict")
            if not resp.get("ok"):
                raise RuntimeError(
                    f"sandbox tool {method}: {resp.get('error', 'unknown error')}"
                )
            # Wrap the dict result so skill code can use BOTH `r.last_frame`
            # and `r["last_frame"]`. Lists/scalars pass through unchanged.
            return _wrap(resp.get("value"))
        return call

    tools_ns: Any = None
    if rpc_enabled:
        tools_ns = SimpleNamespace(take_actions=_make_rpc_tool("take_actions"))

    globals_dict: dict[str, Any] = {
        "__builtins__": _safe_builtins(),
        "state": state,
        "args": args,
        "render_grid": _render_grid_impl,
        "render_grids": _render_grids_impl,
        **safe_modules,
    }
    if tools_ns is not None:
        globals_dict["tools"] = tools_ns

    out_buf, err_buf = io.StringIO(), io.StringIO()
    result_obj: dict[str, Any] = {"type": "result", "success": False}
    try:
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            exec(compiled_code, globals_dict)  # noqa: S102 - intentionally sandboxed
        result_obj["success"] = True
        if "result" in globals_dict:
            try:
                result_obj["result"] = json.loads(
                    json.dumps(globals_dict["result"], default=str)
                )
            except (TypeError, ValueError):
                result_obj["result"] = repr(globals_dict["result"])
    except SystemExit as exc:
        result_obj["error"] = f"SystemExit: {exc.code!r}"
    except MemoryError:
        result_obj["error"] = "MemoryError"
    except Exception as exc:
        result_obj["error"] = f"{type(exc).__name__}: {exc}"

    # No-output safeguard: if the skill ran cleanly but produced no result,
    # no stdout, no stderr, AND made no engine RPC, AND the code contains a
    # function def, it likely defined `def run(args): ...` without calling
    # it. Surface a hint so the model can fix the pattern next turn.
    if (
        result_obj.get("success")
        and "result" not in globals_dict
        and not out_buf.getvalue()
        and not err_buf.getvalue()
        and rpc_invocations[0] == 0
        and "def " in code
    ):
        err_buf.write(
            "WARNING: skill produced no result, no stdout, and made no "
            "engine calls. If you wrapped logic in `def run(args): ...`, "
            "add `result = run(args)` after the def — nothing is invoked "
            "automatically.\n"
        )

    result_obj["stdout"] = out_buf.getvalue()[:STDOUT_CAP]
    result_obj["stderr"] = err_buf.getvalue()[:STDERR_CAP]

    real_stdout.write(json.dumps(result_obj, default=str) + "\n")
    real_stdout.flush()
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        raise SystemExit(_worker_main())
    sys.stderr.write("sandbox.py must be invoked with --worker\n")
    raise SystemExit(2)
