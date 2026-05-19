from __future__ import annotations

# NOTE: deliberately minimal top-level imports. Anything imported here is also
# imported in the worker process (sandbox.py runs as a script via --worker), and
# every module imported in the worker becomes reachable via class-MRO traversal
# (().__class__.__mro__[-1].__subclasses__()). Heavy parent-only modules like
# `subprocess`, `tempfile`, `logging`, `pathlib` are imported INSIDE
# run_python_snippet() so they never load in the worker.
import json
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

# --- Caps (parent-side; child also enforces via rlimit when available) -------

CODE_MAX_CHARS = 8000
ARGS_JSON_MAX_CHARS = 4000
STDOUT_CAP = 5000
STDERR_CAP = 5000
RESULT_CAP = 5000
DEFAULT_TIMEOUT_S = 5.0


# --- Worker-side resource limits (Unix best-effort) --------------------------

RLIMIT_CPU_SECONDS = 5
RLIMIT_AS_BYTES = 256 * 1024 * 1024
RLIMIT_FSIZE_BYTES = 1 * 1024 * 1024
RLIMIT_NOFILE = 32
RLIMIT_NPROC = 1

_BANNED_NAMES = {
    "__builtins__",
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}

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
    "getattr",
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

# Module names blocked at `import` audit time. Note that user code already
# cannot write `import os` (the AST validator rejects import statements). This
# list only protects against lazy/internal imports that might be triggered by
# something we didn't anticipate — stdlib lazy imports like `heapq` (used by
# Counter.most_common) are intentionally NOT here.
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
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Spawn a child Python process, send a JSON payload, read a JSON result.

    Never raises. On any failure returns a dict with success=False and an
    error message. Defense-in-depth — see plan for the full threat model.
    """
    # Parent-only imports stay local so the worker doesn't pull subprocess /
    # tempfile / pathlib into its class-MRO surface.
    import subprocess
    import tempfile
    from pathlib import Path

    if not isinstance(code, str) or not code.strip():
        return {"success": False, "error": "code is empty"}
    if len(code) > CODE_MAX_CHARS:
        return {
            "success": False,
            "error": f"code exceeds {CODE_MAX_CHARS} chars",
        }
    validation_error = _validate_code(code)
    if validation_error is not None:
        return {"success": False, "error": validation_error}

    args = dict(args or {})
    try:
        args_json = json.dumps(args)
    except (TypeError, ValueError) as exc:
        return {
            "success": False,
            "error": f"args not JSON-serializable: {exc}",
        }
    if len(args_json) > ARGS_JSON_MAX_CHARS:
        return {
            "success": False,
            "error": f"args exceed {ARGS_JSON_MAX_CHARS} chars",
        }

    # `default=str` is a defensive fallback: the agent already converts the
    # frame via model_dump(mode="json"), but if any trajectory or memory entry
    # happens to contain a non-JSON value (rare; usually str/int/dict), we
    # stringify it rather than crash the sandbox call.
    try:
        payload = json.dumps(
            {
                "code": code,
                "args": args,
                "state": asdict(state),
            },
            default=str,
        )
    except (TypeError, ValueError) as exc:
        return {
            "success": False,
            "error": f"sandbox payload not JSON-serializable: {exc}",
        }

    worker_script = str(Path(__file__).resolve())
    # -I: ignore PYTHON* env vars and don't add cwd to sys.path.
    # -S: skip site.py and site-packages.
    cmd = [sys.executable, "-I", "-S", worker_script, "--worker"]
    env = {"PYTHONNOUSERSITE": "1", "LANG": "C", "PATH": ""}

    try:
        with tempfile.TemporaryDirectory(prefix="ch-sandbox-") as cwd:
            proc = subprocess.run(
                cmd,
                input=payload,
                text=True,
                capture_output=True,
                cwd=cwd,
                env=env,
                timeout=timeout_s,
                check=False,
                start_new_session=True,
            )
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"sandbox timeout after {timeout_s:.1f}s",
            "stdout": "",
            "stderr": "",
        }
    except OSError as exc:
        return {"success": False, "error": f"sandbox spawn failed: {exc}"}

    stdout = proc.stdout or ""
    stderr = (proc.stderr or "")[:STDERR_CAP]

    # The worker prints exactly one JSON object on its last non-empty line.
    last_line = ""
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            last_line = stripped

    if not last_line:
        return {
            "success": False,
            "error": "sandbox worker did not emit JSON",
            "stdout": stdout[:STDOUT_CAP],
            "stderr": stderr,
        }

    try:
        result = json.loads(last_line)
    except json.JSONDecodeError:
        return {
            "success": False,
            "error": "sandbox worker emitted invalid JSON",
            "stdout": stdout[:STDOUT_CAP],
            "stderr": stderr,
        }

    if not isinstance(result, dict):
        return {
            "success": False,
            "error": "sandbox worker did not return a JSON object",
            "stdout": stdout[:STDOUT_CAP],
            "stderr": stderr,
        }

    # Apply hard caps to every text field the worker may have provided.
    result.setdefault("success", False)
    result["stdout"] = str(result.get("stdout") or "")[:STDOUT_CAP]
    result["stderr"] = str(result.get("stderr") or stderr)[:STDERR_CAP]

    if "result" in result:
        try:
            rendered = json.dumps(result["result"], default=str)
        except (TypeError, ValueError):
            rendered = repr(result["result"])
        if len(rendered) > RESULT_CAP:
            rendered = rendered[: RESULT_CAP - 1] + "…"
            result["result_truncated"] = True
            # Keep the truncated form as a string (it's no longer valid JSON).
            result["result"] = rendered
        else:
            try:
                result["result"] = json.loads(rendered)
            except json.JSONDecodeError:
                result["result"] = rendered

    return result


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
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "sandbox policy: import statements are not allowed"
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
    import builtins as _b

    return {
        name: getattr(_b, name) for name in _SAFE_BUILTIN_NAMES if hasattr(_b, name)
    }


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
                if module_name in _BANNED_IMPORT_MODULES or root in _BANNED_IMPORT_MODULES:
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


def _safe_modules() -> dict[str, Any]:
    import collections
    import copy
    import dataclasses
    import functools
    import hashlib
    import itertools
    import json as _json
    import math
    import random
    import re as _re
    import statistics
    import types as _types

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
    }


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


def _worker_main() -> int:
    import io
    from contextlib import redirect_stderr, redirect_stdout
    from types import SimpleNamespace

    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.stdout.write(json.dumps({"success": False, "error": f"bad input: {exc}"}))
        return 0

    _apply_rlimits()

    code = str(payload.get("code") or "")
    args = dict(payload.get("args") or {})
    state_dict = dict(payload.get("state") or {})

    state = SimpleNamespace(**state_dict)
    validation_error = _validate_code(code)
    if validation_error is not None:
        sys.stdout.write(json.dumps({"success": False, "error": validation_error}))
        return 0

    try:
        compiled_code = compile(code, "<continual-harness-sandbox>", "exec")
    except SyntaxError as exc:
        sys.stdout.write(
            json.dumps({"success": False, "error": f"SyntaxError: {exc.msg}"})
        )
        return 0

    safe_modules = _safe_modules()
    _install_audit_hook(allow_exec_events=1)

    globals_dict: dict[str, Any] = {
        "__builtins__": _safe_builtins(),
        "state": state,
        "args": args,
        **safe_modules,
    }

    out_buf, err_buf = io.StringIO(), io.StringIO()
    result_obj: dict[str, Any] = {"success": False}
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

    result_obj["stdout"] = out_buf.getvalue()[:STDOUT_CAP]
    result_obj["stderr"] = err_buf.getvalue()[:STDERR_CAP]

    sys.stdout.write(json.dumps(result_obj, default=str))
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        raise SystemExit(_worker_main())
    sys.stderr.write("sandbox.py must be invoked with --worker\n")
    raise SystemExit(2)
