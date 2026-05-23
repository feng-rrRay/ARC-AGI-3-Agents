from __future__ import annotations

import platform
import subprocess
from typing import Any

import pytest

from agents.templates.continual_harness.sandbox import (
    CODE_MAX_CHARS,
    RESULT_CAP,
    STDOUT_CAP,
    SandboxState,
    _validate_code,
    run_python_snippet,
)


def _empty_state() -> SandboxState:
    return SandboxState()


@pytest.mark.unit
class TestSandboxBasic:
    def test_simple_snippet_sets_result(self) -> None:
        out = run_python_snippet("result = 1 + 2", state=_empty_state())
        assert out["success"] is True
        assert out["result"] == 3

    def test_print_is_captured(self) -> None:
        out = run_python_snippet('print("hi")\nresult = None', state=_empty_state())
        assert out["success"] is True
        assert "hi" in out["stdout"]

    def test_args_dict_is_bound(self) -> None:
        out = run_python_snippet(
            'result = args["x"] * 2', state=_empty_state(), args={"x": 7}
        )
        assert out["success"] is True
        assert out["result"] == 14

    def test_state_latest_frame_is_visible(self) -> None:
        state = SandboxState(latest_frame={"state": "NOT_FINISHED", "score": 1})
        out = run_python_snippet('result = state.latest_frame["state"]', state=state)
        assert out["success"] is True
        assert out["result"] == "NOT_FINISHED"

    def test_state_recent_trajectory_is_accessible(self) -> None:
        state = SandboxState(
            recent_trajectory=[
                {"action_counter": 1, "chosen_action": "ACTION1"},
                {"action_counter": 2, "chosen_action": "ACTION2"},
            ]
        )
        out = run_python_snippet(
            "result = [r['chosen_action'] for r in state.recent_trajectory]",
            state=state,
        )
        assert out["success"] is True
        assert out["result"] == ["ACTION1", "ACTION2"]

    def test_state_memory_and_skill_entries_are_visible(self) -> None:
        state = SandboxState(
            memory_entries=[{"id": "mem_001", "title": "x"}],
            skill_entries=[{"id": "skill_001", "name": "f"}],
        )
        out = run_python_snippet(
            "result = [state.memory_entries[0]['id'], state.skill_entries[0]['id']]",
            state=state,
        )
        assert out["success"] is True
        assert out["result"] == ["mem_001", "skill_001"]


@pytest.mark.unit
class TestSandboxSafety:
    def test_import_os_is_blocked(self) -> None:
        out = run_python_snippet("import os\nresult = 1", state=_empty_state())
        assert out["success"] is False
        assert "sandbox policy" in (out.get("error") or "")

    def test_open_is_blocked(self) -> None:
        out = run_python_snippet(
            'open("/tmp/x", "w")\nresult = 1', state=_empty_state()
        )
        assert out["success"] is False
        assert "sandbox policy" in (out.get("error") or "")

    def test_eval_is_blocked(self) -> None:
        out = run_python_snippet('result = eval("1+1")', state=_empty_state())
        assert out["success"] is False
        assert "sandbox policy" in (out.get("error") or "")

    def test_exec_is_blocked(self) -> None:
        out = run_python_snippet('exec("x = 1")\nresult = 1', state=_empty_state())
        assert out["success"] is False
        assert "sandbox policy" in (out.get("error") or "")

    def test_compile_is_blocked(self) -> None:
        out = run_python_snippet(
            'result = compile("1", "<x>", "eval")', state=_empty_state()
        )
        assert out["success"] is False
        assert "sandbox policy" in (out.get("error") or "")

    def test_class_mro_does_not_expose_popen(self) -> None:
        # The classic class-mro escape is rejected before execution.
        code = (
            "subs = ().__class__.__mro__[-1].__subclasses__()\n"
            "names = [c.__name__ for c in subs]\n"
            "result = 'Popen' in names\n"
        )
        out = run_python_snippet(code, state=_empty_state())
        assert out["success"] is False
        assert "sandbox policy" in (out.get("error") or "")

    def test_class_mro_does_not_expose_socket(self) -> None:
        code = (
            "subs = ().__class__.__mro__[-1].__subclasses__()\n"
            "names = [c.__name__ for c in subs]\n"
            "result = 'socket' in [n.lower() for n in names]\n"
        )
        out = run_python_snippet(code, state=_empty_state())
        assert out["success"] is False
        assert "sandbox policy" in (out.get("error") or "")

    def test_catch_warnings_import_escape_is_blocked(self) -> None:
        code = (
            "for cls in object.__subclasses__():\n"
            "    if cls.__name__ == 'catch_warnings':\n"
            "        builtins = cls()._module.__builtins__\n"
            "        os = builtins['__import__']('os')\n"
            "        result = os.listdir('/')\n"
        )
        out = run_python_snippet(code, state=_empty_state())
        assert out["success"] is False
        assert "sandbox policy" in (out.get("error") or "")

    def test_private_module_attribute_escape_is_blocked(self) -> None:
        out = run_python_snippet(
            "result = random._os.listdir('/')", state=_empty_state()
        )
        assert out["success"] is False
        assert "sandbox policy" in (out.get("error") or "")

    def test_curated_modules_do_not_expose_inspect_escape_path(self) -> None:
        out = run_python_snippet("result = dataclasses.inspect", state=_empty_state())
        assert out["success"] is False
        assert "AttributeError" in (out.get("error") or "")

    def test_curated_modules_do_not_expose_json_decoder_graph(self) -> None:
        out = run_python_snippet("result = json.decoder", state=_empty_state())
        assert out["success"] is False
        assert "AttributeError" in (out.get("error") or "")

    def test_re_compile_remains_available(self) -> None:
        out = run_python_snippet(
            "result = re.compile('x').match('x') is not None",
            state=_empty_state(),
        )
        assert out["success"] is True
        assert out["result"] is True

    def test_zero_division_caught(self) -> None:
        out = run_python_snippet("result = 1 / 0", state=_empty_state())
        assert out["success"] is False
        assert "ZeroDivisionError" in (out.get("error") or "")

    def test_syntax_error_caught(self) -> None:
        out = run_python_snippet("def (", state=_empty_state())
        assert out["success"] is False
        assert "SyntaxError" in (out.get("error") or "")


@pytest.mark.unit
class TestSandboxImportWhitelist:
    """The AST validator allows `import X` only when X is pre-loaded; the
    audit hook still blocks dangerous root modules at runtime as a backstop.
    """

    def test_validator_accepts_import_numpy_as_np(self) -> None:
        assert _validate_code("import numpy as np\nresult = 0") is None

    def test_validator_accepts_from_collections_import(self) -> None:
        assert (
            _validate_code("from collections import Counter\nresult = 0") is None
        )

    def test_validator_accepts_from_pil_import(self) -> None:
        assert (
            _validate_code("from PIL import Image, ImageDraw\nresult = 0") is None
        )

    def test_validator_accepts_multi_module_import(self) -> None:
        assert _validate_code("import numpy, collections\nresult = 0") is None

    def test_validator_rejects_import_os(self) -> None:
        err = _validate_code("import os")
        assert err is not None and "not allowed" in err

    def test_validator_rejects_from_os_import(self) -> None:
        err = _validate_code("from os import path")
        assert err is not None and "not allowed" in err

    def test_validator_rejects_mixed_safe_and_unsafe(self) -> None:
        err = _validate_code("import numpy, os")
        assert err is not None and "not allowed" in err

    def test_validator_rejects_relative_import(self) -> None:
        err = _validate_code("from . import x")
        assert err is not None and "relative" in err

    def test_runtime_import_numpy_succeeds_end_to_end(self) -> None:
        out = run_python_snippet(
            "import numpy as np\nresult = int(np.array([1, 2, 3]).sum())",
            state=_empty_state(),
        )
        assert out["success"] is True
        assert out["result"] == 6


@pytest.mark.unit
class TestSandboxModules:
    def test_whitelisted_math_works(self) -> None:
        out = run_python_snippet("result = math.sqrt(16)", state=_empty_state())
        assert out["success"] is True
        assert out["result"] == 4.0

    def test_whitelisted_json_works(self) -> None:
        out = run_python_snippet(
            "result = json.loads('{\"a\": 1}')", state=_empty_state()
        )
        assert out["success"] is True
        assert out["result"] == {"a": 1}

    def test_blocklisted_pathlib_not_visible(self) -> None:
        out = run_python_snippet('result = pathlib.Path("/")', state=_empty_state())
        assert out["success"] is False
        assert "NameError" in (out.get("error") or "")


@pytest.mark.unit
class TestSandboxCaps:
    def test_oversize_code_rejected_on_parent(self) -> None:
        big = "x = 1\n" * 5000  # ~30000 chars
        assert len(big) > CODE_MAX_CHARS
        out = run_python_snippet(big, state=_empty_state())
        assert out["success"] is False
        assert "exceeds" in (out.get("error") or "")

    def test_empty_code_rejected(self) -> None:
        out = run_python_snippet("", state=_empty_state())
        assert out["success"] is False
        assert "empty" in (out.get("error") or "")

    def test_args_must_be_json_serializable(self) -> None:
        out = run_python_snippet(
            "result = 1", state=_empty_state(), args={"x": object()}
        )
        assert out["success"] is False
        assert "JSON" in (out.get("error") or "")

    def test_stdout_cap_enforced(self) -> None:
        # Print enough to exceed STDOUT_CAP; sandbox should truncate.
        n = STDOUT_CAP + 1000
        out = run_python_snippet(
            f'print("x" * {n})\nresult = None', state=_empty_state()
        )
        assert out["success"] is True
        assert len(out["stdout"]) <= STDOUT_CAP

    def test_result_cap_enforced_with_truncation_marker(self) -> None:
        n = RESULT_CAP + 1000
        out = run_python_snippet(f'result = "x" * {n}', state=_empty_state())
        assert out["success"] is True
        # When the rendered result blows the cap, parent flips a flag and stores
        # the truncated form.
        assert out.get("result_truncated") is True
        assert isinstance(out["result"], str)
        assert len(out["result"]) <= RESULT_CAP


@pytest.mark.unit
class TestSandboxTimeout:
    def test_busy_loop_terminated_by_timeout(self) -> None:
        # 0.5s timeout: a tight Python loop should be killed by the parent's
        # subprocess.run timeout if the worker's RLIMIT_CPU doesn't catch it first.
        out = run_python_snippet(
            "while True: pass", state=_empty_state(), timeout_s=0.5
        )
        assert out["success"] is False
        msg = (out.get("error") or "").lower()
        assert "timeout" in msg or "exceeded" in msg or "cpu" in msg


@pytest.mark.unit
class TestSandboxResourceLimits:
    @pytest.mark.skipif(
        platform.system() == "Windows", reason="resource module is POSIX-only"
    )
    def test_memory_bomb_caught_or_rejected(self) -> None:
        # RLIMIT_AS should make a 1 GB allocation fail. On macOS this may not
        # always trigger; accept either MemoryError or any other failure.
        out = run_python_snippet(
            "result = bytearray(10**9)", state=_empty_state(), timeout_s=3.0
        )
        # We tolerate success on platforms where rlimit isn't enforced — but on
        # Linux (CI) this will reliably fail.
        if out["success"]:
            pytest.skip("rlimit not enforced on this platform")
        # When it fails we want a clean error string, not a hung process.
        assert out.get("error")


@pytest.mark.unit
class TestSandboxErrorPaths:
    def test_invalid_worker_output_returns_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the subprocess emits garbage instead of JSON, parent stays clean."""

        def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess:
            return subprocess.CompletedProcess(
                args=args[0] if args else [],
                returncode=0,
                stdout="this is not json at all",
                stderr="",
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
        out = run_python_snippet("result = 1", state=_empty_state())
        assert out["success"] is False
        assert "did not emit JSON" in (out.get("error") or "")


@pytest.mark.unit
class TestSandboxReflection:
    """Fix A — `getattr`/`hasattr` are public; `heapq` is pre-loaded."""

    def test_getattr_allowed_at_parse(self) -> None:
        assert _validate_code('x = getattr(state, "latest_frame", None)') is None

    def test_hasattr_allowed_at_parse(self) -> None:
        assert _validate_code('x = hasattr(state, "latest_frame")') is None

    def test_getattr_with_dunder_string_still_rejected(self) -> None:
        # The dunder-string check fires before getattr is involved.
        err = _validate_code('x = getattr(o, "__class__")')
        assert err is not None and "dunder strings" in err

    def test_setattr_still_rejected(self) -> None:
        err = _validate_code('setattr(o, "x", 1)')
        assert err is not None and "setattr" in err

    def test_getattr_runs_end_to_end(self) -> None:
        state = SandboxState(latest_frame={"state": "NOT_FINISHED"})
        out = run_python_snippet(
            'result = [hasattr(state, "latest_frame"), '
            'getattr(state, "missing", "fallback")]',
            state=state,
        )
        assert out["success"] is True
        assert out["result"] == [True, "fallback"]

    def test_heapq_import_allowed(self) -> None:
        assert _validate_code("import heapq") is None
        assert _validate_code("from heapq import heappush, heappop") is None

    def test_heapq_runs_end_to_end(self) -> None:
        out = run_python_snippet(
            "import heapq\nh = []\n"
            "for v in (5, 1, 3, 2, 4):\n    heapq.heappush(h, v)\n"
            "result = [heapq.heappop(h) for _ in range(len(h))]",
            state=_empty_state(),
        )
        assert out["success"] is True
        assert out["result"] == [1, 2, 3, 4, 5]


@pytest.mark.unit
class TestSandboxDualAccess:
    """Fix C — state and RPC return values accept both attr and subscript."""

    def test_state_top_level_attr_and_subscript(self) -> None:
        state = SandboxState(latest_frame={"a": 1, "b": 2})
        out = run_python_snippet(
            "result = [state.latest_frame, state['latest_frame']]",
            state=state,
        )
        assert out["success"] is True
        assert out["result"][0] == {"a": 1, "b": 2}
        assert out["result"][1] == {"a": 1, "b": 2}

    def test_state_nested_dict_dual_access(self) -> None:
        state = SandboxState(latest_frame={"frame": [[0, 1], [2, 3]]})
        out = run_python_snippet(
            "result = [state.latest_frame.frame, "
            "state['latest_frame']['frame'], "
            "state.latest_frame['frame'], "
            "state['latest_frame'].frame]",
            state=state,
        )
        assert out["success"] is True
        first = out["result"][0]
        assert all(part == first for part in out["result"])

    def test_dict_inside_list_is_also_wrapped(self) -> None:
        state = SandboxState(
            recent_trajectory=[{"step": 1, "action": "ACTION1"}]
        )
        out = run_python_snippet(
            "row = state.recent_trajectory[0]\n"
            "result = [row.step, row['step'], row.action]",
            state=state,
        )
        assert out["success"] is True
        assert out["result"] == [1, 1, "ACTION1"]

    def test_missing_key_raises_attribute_error(self) -> None:
        state = SandboxState(latest_frame={"a": 1})
        out = run_python_snippet(
            "try:\n"
            "    _ = state.latest_frame.nonexistent\n"
            "    result = 'no-error'\n"
            "except AttributeError:\n"
            "    result = 'attribute-error'\n",
            state=state,
        )
        assert out["success"] is True
        assert out["result"] == "attribute-error"
