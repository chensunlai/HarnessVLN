from __future__ import annotations

import asyncio
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from agents import PassthroughVLNAgent
from envs import DummyNavigationEnvironment
from harness import NavigationHarness, NavigationStack
from harness.media import FileArrayStore
from harness.tool_bus import Tool, ToolBus
from schemas import NavGoal, NavTask
from vln.rpc import JsonLineProcess, RPCError, RPCVLNNavigator


WORKER = Path(__file__).resolve().parents[1] / "fixtures" / "rpc_worker.py"
SDK_WORKER = Path(__file__).resolve().parents[1] / "fixtures" / "sdk_worker.py"
ROOT = Path(__file__).resolve().parents[2]


def test_jsonl_worker_can_call_scoped_navigation_tools() -> None:
    async def scenario():
        calls = []

        async def observe(actor, arguments):
            calls.append((actor, arguments))
            return {"observation_id": "1", "channels": {"rgb": "ref://rgb"}}

        bus = ToolBus()
        bus.register(
            (
                Tool(
                    "nav.observe",
                    "Observe.",
                    {"type": "object", "additionalProperties": False},
                    observe,
                ),
            )
        )
        process = JsonLineProcess((sys.executable, str(WORKER)), request_timeout_s=1)
        hello = await process.start(
            bus.client("vln", frozenset({"nav.observe"})),
            {"protocol": 1, "model": "probe"},
        )
        result = await process.request(
            "probe_tool", {"name": "nav.observe", "arguments": {}}
        )
        await process.close()

        assert hello == {"protocol": 1, "model": "probe"}
        assert result["channels"]["rgb"] == "ref://rgb"
        assert calls == [("vln", {})]
        assert bus.audit[0].actor == "vln"

    asyncio.run(scenario())


def test_start_cancellation_after_spawn_reaps_child_and_closes_sockets(
    monkeypatch,
) -> None:
    async def scenario():
        real_spawn = asyncio.create_subprocess_exec
        real_socketpair = socket.socketpair
        spawned = asyncio.Event()
        release_spawn = asyncio.Event()
        child_pid = None
        sockets = []

        def capture_socketpair(*args, **kwargs):
            pair = real_socketpair(*args, **kwargs)
            sockets.extend(pair)
            return pair

        async def delayed_spawn(*args, **kwargs):
            nonlocal child_pid
            child = await real_spawn(*args, **kwargs)
            child_pid = child.pid
            spawned.set()
            await release_spawn.wait()
            return child

        monkeypatch.setattr(socket, "socketpair", capture_socketpair)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_spawn)
        process = JsonLineProcess(
            (sys.executable, "-c", "import time; time.sleep(60)"),
            request_timeout_s=1,
        )
        bus = ToolBus()
        starting = asyncio.create_task(
            process.start(
                bus.client("vln", frozenset()),
                {"protocol": 1, "model": "x"},
            )
        )
        await spawned.wait()

        starting.cancel()
        await asyncio.sleep(0)
        release_spawn.set()

        with pytest.raises(asyncio.CancelledError):
            await starting
        assert child_pid is not None
        assert process.returncode is not None
        assert process.terminated
        assert len(sockets) >= 2
        assert sockets[0].fileno() == -1
        assert sockets[1].fileno() == -1

    asyncio.run(scenario())


@pytest.mark.parametrize("transient_fence_failure", [False, True])
def test_close_kills_worker_descendants_after_leader_already_exited(
    monkeypatch, transient_fence_failure
) -> None:
    async def scenario():
        leader_code = (
            "import subprocess,sys; "
            "child=subprocess.Popen([sys.executable,'-c',"
            "'import time; time.sleep(60)'], stdin=subprocess.DEVNULL, "
            "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
            "print(child.pid, flush=True)"
        )
        leader = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            leader_code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        assert leader.stdout is not None
        descendant_pid = int((await leader.stdout.readline()).decode().strip())
        assert await leader.wait() == 0
        transport = JsonLineProcess(("unused",))
        transport._process = leader
        fence_calls = 0
        if transient_fence_failure:
            real_fence = transport._fence_process_group

            def flaky_fence(process):
                nonlocal fence_calls
                fence_calls += 1
                if fence_calls == 1:
                    raise OSError("transient process-group fence failure")
                real_fence(process)

            monkeypatch.setattr(transport, "_fence_process_group", flaky_fence)

        if transient_fence_failure:
            with pytest.raises(OSError, match="transient process-group fence failure"):
                await transport.close()
            assert fence_calls == 2
            await transport.close()
        else:
            await transport.close()

        deadline = asyncio.get_running_loop().time() + 1
        while asyncio.get_running_loop().time() < deadline:
            try:
                state = Path(f"/proc/{descendant_pid}/stat").read_text().split()[2]
            except FileNotFoundError:
                break
            if state == "Z":
                break
            await asyncio.sleep(0.01)
        else:
            os.killpg(leader.pid, 9)
            pytest.fail("worker descendant survived process-group cleanup")
        assert transport.quiescent

    asyncio.run(scenario())


def test_rpc_navigator_exposes_one_blocking_task_tool(tmp_path) -> None:
    async def scenario():
        checkpoint = tmp_path / "checkpoint"
        checkpoint.write_text("fixture")
        goal = NavGoal("goal", "already at target")
        navigator = RPCVLNNavigator(
            (sys.executable, str(WORKER)),
            upstream_root=tmp_path,
            checkpoint=checkpoint,
            request_timeout_s=1,
        )
        result = await NavigationHarness(timeout_s=2).run_task(
            NavTask("rpc", goal),
            NavigationStack(
                PassthroughVLNAgent(),
                DummyNavigationEnvironment((goal,), targets=(0,)),
                vln=navigator,
            ),
        )
        assert result.terminal.status == "completed"
        assert [event.name for event in result.audit].count("vln.navigate.task") == 1
        assert not any(
            event.name in {
                "vln.navigate.start",
                "vln.navigate.status",
                "vln.navigate.cancel",
            }
            for event in result.audit
        )

    asyncio.run(scenario())


def test_rpc_local_call_uses_requested_bounded_budget(tmp_path, monkeypatch) -> None:
    async def scenario():
        navigator = RPCVLNNavigator(
            ("worker",),
            upstream_root=tmp_path,
            checkpoint=tmp_path / "checkpoint",
            local_max_steps=16,
        )
        captured = {}

        async def run_blocking(actor, instruction, options):
            captured.update(actor=actor, instruction=instruction, options=dict(options))
            return {"state": "limit_reached", "steps": 4, "reason": "limit"}

        monkeypatch.setattr(navigator, "_run_blocking_job", run_blocking)

        result = await navigator._navigate_local(
            "agent", {"instruction": "Stop beside the visible chair.", "max_steps": 4}
        )

        assert result["steps"] == 4
        assert captured == {
            "actor": "agent",
            "instruction": "Stop beside the visible chair.",
            "options": {"max_steps": 4},
        }

    asyncio.run(scenario())


@pytest.mark.parametrize("value", [True, 1.5, 0, -1])
def test_rpc_local_step_limit_requires_positive_integer(tmp_path, value) -> None:
    with pytest.raises(ValueError, match="local_max_steps"):
        RPCVLNNavigator(
            ("worker",),
            upstream_root=tmp_path,
            checkpoint=tmp_path / "checkpoint",
            local_max_steps=value,
        )


def test_rpc_blocking_result_does_not_expose_worker_traceback(
    tmp_path, monkeypatch
) -> None:
    async def scenario():
        navigator = RPCVLNNavigator(
            ("worker",),
            upstream_root=tmp_path,
            checkpoint=tmp_path / "checkpoint",
        )

        async def start_job(actor, arguments):
            return {"job_id": "job"}

        async def status_job(actor, arguments):
            return {
                "job_id": "job",
                "state": "failed",
                "reason": "RuntimeError: inference failed",
                "traceback": "internal worker traceback",
            }

        monkeypatch.setattr(navigator, "_start_job", start_job)
        monkeypatch.setattr(navigator, "_status_job", status_job)

        result = await navigator._run_blocking_job("agent", "local", {})

        assert result == {
            "state": "failed",
            "reason": "RuntimeError: inference failed",
        }

    asyncio.run(scenario())


def test_worker_sdk_runs_model_owned_navigation_loop(tmp_path) -> None:
    class SDKNavigator(RPCVLNNavigator):
        model_name = "sdk-fixture"
        required_tools = frozenset({"nav.observe", "nav.move.discrete"})

    async def scenario():
        class ArrayEnvironment(DummyNavigationEnvironment):
            async def _observe(self, actor, arguments):
                observation = await super()._observe(actor, arguments)
                observation["channels"]["rgb"] = np.arange(12, dtype=np.uint8).reshape(
                    2, 2, 3
                )
                return observation

        checkpoint = tmp_path / "checkpoint"
        checkpoint.write_text("fixture")
        goal = NavGoal("goal", "move to the target")
        navigator = SDKNavigator(
            (sys.executable, str(SDK_WORKER)),
            upstream_root=tmp_path,
            checkpoint=checkpoint,
            env={"PYTHONPATH": str(ROOT / "src")},
            worker_options={"max_steps": 8},
            request_timeout_s=1,
        )
        result = await NavigationHarness(timeout_s=2).run_task(
            NavTask("sdk", goal),
            NavigationStack(
                PassthroughVLNAgent(),
                ArrayEnvironment((goal,), targets=(2,)),
                vln=navigator,
            ),
        )

        assert result.terminal.status == "completed"
        assert result.environment["position"] == 2
        assert [
            event.actor for event in result.audit if event.name == "nav.observe"
        ] == [
            "vln",
            "vln",
            "vln",
        ]
        assert [event.name for event in result.audit if event.actor == "vln"].count(
            "nav.move.discrete"
        ) == 2

    asyncio.run(scenario())


def test_session_scoped_worker_is_reused_and_releases_each_jobs_media(tmp_path) -> None:
    class SDKNavigator(RPCVLNNavigator):
        model_name = "sdk-fixture"
        required_tools = frozenset({"nav.observe", "nav.move.discrete"})

    class ArrayEnvironment(DummyNavigationEnvironment):
        async def _observe(self, actor, arguments):
            observation = await super()._observe(actor, arguments)
            observation["channels"]["rgb"] = np.arange(12, dtype=np.uint8).reshape(
                2, 2, 3
            )
            return observation

    async def scenario():
        checkpoint = tmp_path / "checkpoint"
        checkpoint.write_text("fixture")
        navigator = SDKNavigator(
            (sys.executable, str(SDK_WORKER)),
            upstream_root=tmp_path,
            checkpoint=checkpoint,
            env={"PYTHONPATH": str(ROOT / "src")},
            worker_options={"max_steps": 8},
            request_timeout_s=1,
        )
        navigator.enable_session_scope()
        process = None
        process_id = None
        media_root = None

        for index in range(2):
            goal = NavGoal(f"goal-{index}", f"task {index}")
            result = await NavigationHarness(timeout_s=2).run_task(
                NavTask(f"task-{index}", goal),
                NavigationStack(
                    PassthroughVLNAgent(),
                    ArrayEnvironment((goal,), targets=(2,)),
                    vln=navigator,
                ),
            )
            assert result.terminal.status == "completed"
            assert navigator._process is not None
            if process is None:
                process = navigator._process
                assert process._process is not None
                process_id = process._process.pid
            assert navigator._process is process
            assert navigator._process._process is not None
            assert navigator._process._process.pid == process_id
            media_root = navigator._process._media.root
            assert media_root is not None and media_root.is_dir()
            assert list(media_root.iterdir()) == []

        assert media_root is not None
        await navigator.close_session()
        assert not media_root.exists()
        assert navigator._process is None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure_flag", ["--drop-first-start", "--malformed-first-start"]
)
def test_ambiguous_start_discards_worker_before_next_task(
    tmp_path, failure_flag
) -> None:
    async def scenario():
        checkpoint = tmp_path / "checkpoint"
        checkpoint.write_text("fixture")
        navigator = RPCVLNNavigator(
            (sys.executable, str(WORKER), failure_flag),
            upstream_root=tmp_path,
            checkpoint=checkpoint,
            request_timeout_s=0.05,
        )
        navigator.enable_session_scope()
        bus = ToolBus()
        await navigator.start(
            NavTask("first", NavGoal("goal-1", "first")),
            bus.client("vln", frozenset()),
        )
        old_process = navigator._process
        assert old_process is not None and old_process._process is not None
        old_pid = old_process._process.pid

        with pytest.raises(RPCError):
            await navigator._start_job("agent", {"instruction": "first", "options": {}})

        assert navigator._process is None
        assert old_process.returncode is not None

        navigator.command = (sys.executable, str(WORKER))
        await navigator.start(
            NavTask("second", NavGoal("goal-2", "second")),
            bus.client("vln", frozenset()),
        )
        started = await navigator._start_job(
            "agent", {"instruction": "second", "options": {}}
        )
        terminal = await navigator._status_job("agent", {"job_id": started["job_id"]})
        assert terminal["state"] == "succeeded"
        assert navigator._process is not None
        assert navigator._process._process is not None
        assert navigator._process._process.pid != old_pid
        await navigator.stop("done")
        await navigator.close_session()

    asyncio.run(scenario())


def test_concurrent_status_and_cancel_finalize_job_once(tmp_path) -> None:
    async def scenario():
        checkpoint = tmp_path / "checkpoint"
        checkpoint.write_text("fixture")
        navigator = RPCVLNNavigator(
            (sys.executable, str(WORKER)),
            upstream_root=tmp_path,
            checkpoint=checkpoint,
            request_timeout_s=1,
        )
        navigator.enable_session_scope()
        bus = ToolBus()
        await navigator.start(
            NavTask("race", NavGoal("goal", "race")),
            bus.client("vln", frozenset()),
        )
        started = await navigator._start_job(
            "agent", {"instruction": "race", "options": {}}
        )
        job_id = started["job_id"]

        status, cancelled = await asyncio.gather(
            navigator._status_job("agent", {"job_id": job_id}),
            navigator._cancel_job("harness", {"job_id": job_id}),
        )

        assert status == cancelled
        assert job_id not in navigator._active_jobs
        assert navigator._process is not None and navigator._process.active
        await navigator.stop("done")
        await navigator.close_session()

    asyncio.run(scenario())


def test_late_tool_call_from_released_job_never_reaches_task_tools(tmp_path) -> None:
    async def scenario():
        checkpoint = tmp_path / "checkpoint"
        checkpoint.write_text("fixture")
        calls = []

        async def observe(actor, arguments):
            calls.append((actor, arguments))
            return {}

        bus = ToolBus()
        bus.register(
            (
                Tool(
                    "nav.observe",
                    "Observe.",
                    {"type": "object", "additionalProperties": False},
                    observe,
                ),
            )
        )
        navigator = RPCVLNNavigator(
            (sys.executable, str(WORKER), "--late-tool-after-release"),
            upstream_root=tmp_path,
            checkpoint=checkpoint,
            request_timeout_s=1,
        )
        navigator.enable_session_scope()
        await navigator.start(
            NavTask("late", NavGoal("goal", "late")),
            bus.client("vln", frozenset({"nav.observe"})),
        )
        started = await navigator._start_job(
            "agent", {"instruction": "late", "options": {}}
        )

        try:
            await navigator._status_job("agent", {"job_id": started["job_id"]})
        except RPCError:
            pass

        process = navigator._process
        if process is not None:
            deadline = asyncio.get_running_loop().time() + 1
            while process.active and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.001)
            assert not process.active

        assert calls == []
        await navigator.close_session()
        assert navigator._process is None

    asyncio.run(scenario())


def test_process_close_has_bounded_shutdown_deadline() -> None:
    async def scenario():
        bus = ToolBus()
        process = JsonLineProcess(
            (sys.executable, str(WORKER), "--ignore-shutdown"),
            request_timeout_s=30,
        )
        await process.start(
            bus.client("vln", frozenset()), {"protocol": 1, "model": "x"}
        )

        started = time.monotonic()
        await process.close()

        assert time.monotonic() - started < 3
        assert process.returncode is not None

    asyncio.run(scenario())


def test_graceful_shutdown_waits_for_worker_backend_close(tmp_path) -> None:
    async def scenario():
        marker = tmp_path / "backend-closed"
        process = JsonLineProcess(
            (sys.executable, str(SDK_WORKER)),
            env={
                "PYTHONPATH": str(ROOT / "src"),
                "HARNESS_CLOSE_MARKER": str(marker),
            },
            request_timeout_s=1,
        )
        bus = ToolBus()
        await process.start(
            bus.client("vln", frozenset()),
            {
                "protocol": 2,
                "model": "sdk-fixture",
                "checkpoint": "fixture",
                "options": {"max_steps": 8},
            },
        )

        await process.close()

        assert process.returncode == 0
        assert marker.read_text(encoding="utf-8") == "closed"

    asyncio.run(scenario())


def test_graceful_shutdown_reports_worker_backend_close_failure(tmp_path) -> None:
    async def scenario():
        process = JsonLineProcess(
            (sys.executable, str(SDK_WORKER)),
            env={
                "PYTHONPATH": str(ROOT / "src"),
                "HARNESS_CLOSE_ERROR": "1",
            },
            request_timeout_s=1,
        )
        bus = ToolBus()
        await process.start(
            bus.client("vln", frozenset()),
            {
                "protocol": 2,
                "model": "sdk-fixture",
                "checkpoint": "fixture",
                "options": {"max_steps": 8},
            },
        )

        with pytest.raises(
            RPCError, match="worker shutdown failed with code 1"
        ) as caught:
            await process.close()

        assert "fixture backend close failed" in str(caught.value)
        assert process.returncode == 1
        assert process.quiescent

    asyncio.run(scenario())


def test_cancelled_reaper_still_kills_and_reaps_worker() -> None:
    async def scenario():
        bus = ToolBus()
        process = JsonLineProcess(
            (
                sys.executable,
                str(WORKER),
                "--ignore-shutdown",
                "--hang-after-hello",
            ),
            request_timeout_s=30,
        )
        await process.start(
            bus.client("vln", frozenset()), {"protocol": 1, "model": "x"}
        )
        media_root = process._media.root
        process._media.encode(np.zeros((2, 2), dtype=np.uint8))
        media_root = process._media.root

        closing = asyncio.create_task(process.close())
        while process._close_task is None:
            await asyncio.sleep(0)
        await asyncio.sleep(0.01)
        process._close_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await closing
        assert process.returncode is not None
        assert media_root is not None and not media_root.exists()

    asyncio.run(scenario())


def test_failed_close_can_retry_process_reaping() -> None:
    async def scenario():
        bus = ToolBus()
        process = JsonLineProcess(
            (sys.executable, str(WORKER), "--ignore-shutdown"),
            request_timeout_s=1,
        )
        await process.start(
            bus.client("vln", frozenset()), {"protocol": 1, "model": "x"}
        )

        async def fail_before_cleanup():
            raise RuntimeError("first close failed")

        process._close_impl = fail_before_cleanup
        with pytest.raises(RuntimeError, match="first close failed"):
            await process.close()
        assert process.returncode is not None

        await process.close()
        assert process.returncode is not None

    asyncio.run(scenario())


def test_process_close_bounds_reverse_tool_cancellation(tmp_path) -> None:
    async def scenario():
        entered = asyncio.Event()
        release_after_cancel = asyncio.Event()

        async def observe(actor, arguments):
            del actor, arguments
            entered.set()
            while not release_after_cancel.is_set():
                try:
                    await release_after_cancel.wait()
                except asyncio.CancelledError:
                    continue

        bus = ToolBus()
        bus.register(
            (
                Tool(
                    "nav.observe",
                    "Observe.",
                    {"type": "object", "additionalProperties": False},
                    observe,
                ),
            )
        )
        process = JsonLineProcess((sys.executable, str(WORKER)), request_timeout_s=30)
        await process.start(
            bus.client("vln", frozenset({"nav.observe"})),
            {"protocol": 1, "model": "x"},
        )
        process._media = FileArrayStore(tmp_path)
        process._media.encode(np.zeros((2, 2), dtype=np.uint8))
        media_root = process._media.root
        request = asyncio.create_task(
            process.request("probe_tool", {"name": "nav.observe", "arguments": {}})
        )
        await entered.wait()

        started = time.monotonic()
        await process.close()

        assert time.monotonic() - started < 3
        assert media_root is not None and not media_root.exists()
        assert process.returncode is not None
        await asyncio.gather(request, return_exceptions=True)
        assert not process.quiescent

        checkpoint = tmp_path / "checkpoint"
        checkpoint.write_text("fixture")
        navigator = RPCVLNNavigator(
            (sys.executable, str(WORKER)),
            upstream_root=tmp_path,
            checkpoint=checkpoint,
            request_timeout_s=1,
        )
        navigator._retired_process = process
        with pytest.raises(RPCError, match="reverse tool calls in quarantine"):
            await navigator.start(
                NavTask("next", NavGoal("goal", "next")),
                bus.client("vln", frozenset({"nav.observe"})),
            )

        release_after_cancel.set()
        deadline = asyncio.get_running_loop().time() + 1
        while not process.quiescent and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.001)
        assert process.quiescent
        await navigator.close_session()
        assert navigator._retired_process is None

    asyncio.run(scenario())


def test_process_close_preserves_cancellation_when_cleanup_fails() -> None:
    async def scenario():
        process = JsonLineProcess(("worker",))
        entered = asyncio.Event()
        release = asyncio.Event()

        async def fail_close():
            entered.set()
            await release.wait()
            raise RuntimeError("cleanup failed")

        process._close_impl = fail_close
        closing = asyncio.create_task(process.close())
        await entered.wait()
        closing.cancel()
        await asyncio.sleep(0)
        assert not closing.done()
        release.set()

        with pytest.raises(asyncio.CancelledError):
            await closing
        assert process._close_task is not None
        assert isinstance(process._close_task.exception(), RuntimeError)

    asyncio.run(scenario())


def test_cleanup_failure_keeps_retired_worker_and_preserves_primary_error(
    tmp_path,
) -> None:
    class FailingProcess:
        active = False
        terminated = False
        quiescent = False

        async def close(self):
            raise RuntimeError("cleanup failed")

    async def scenario():
        checkpoint = tmp_path / "checkpoint"
        checkpoint.write_text("fixture")
        navigator = RPCVLNNavigator(
            ("worker",), upstream_root=tmp_path, checkpoint=checkpoint
        )
        process = FailingProcess()
        navigator._process = process
        primary = RPCError("primary failure")

        with pytest.raises(RPCError, match="primary failure") as caught:
            await navigator._close_after_error_locked(primary)

        assert caught.value is primary
        assert isinstance(caught.value.__cause__, RuntimeError)
        assert navigator._process is None
        assert navigator._retired_process is process

    asyncio.run(scenario())


def test_model_stdout_and_stderr_are_isolated_from_protocol() -> None:
    async def scenario():
        bus = ToolBus()
        process = JsonLineProcess(
            (sys.executable, str(WORKER), "--print-logs"), request_timeout_s=1
        )
        hello = await process.start(
            bus.client("vln", frozenset()), {"protocol": 1, "model": "x"}
        )
        await process.close()
        assert hello == {"protocol": 1, "model": "x"}
        assert "model log leaked to protocol stdout" in process.stdout_tail
        assert "model diagnostic" in process.stderr_tail

    asyncio.run(scenario())


def test_protocol_rejects_invalid_socket_payload() -> None:
    async def scenario():
        bus = ToolBus()
        process = JsonLineProcess(
            (sys.executable, str(WORKER), "--bad-protocol"), request_timeout_s=1
        )
        with pytest.raises(RPCError, match="invalid JSONL"):
            await process.start(
                bus.client("vln", frozenset()), {"protocol": 1, "model": "x"}
            )
        assert process.returncode is not None

    asyncio.run(scenario())


def test_hello_timeout_reaps_process() -> None:
    async def scenario():
        bus = ToolBus()
        process = JsonLineProcess(
            (sys.executable, str(WORKER), "--delay-hello"), request_timeout_s=0.01
        )
        with pytest.raises(RPCError, match="timed out"):
            await process.start(
                bus.client("vln", frozenset()), {"protocol": 1, "model": "x"}
            )
        assert process.returncode is not None

    asyncio.run(scenario())


def test_socket_setup_failure_reaps_process(tmp_path, monkeypatch) -> None:
    async def fail_connection(*args, **kwargs):
        del args, kwargs
        raise OSError("fixture socket setup failed")

    async def scenario():
        bus = ToolBus()
        process = JsonLineProcess((sys.executable, str(WORKER)), request_timeout_s=1)
        monkeypatch.setattr(asyncio, "open_connection", fail_connection)

        with pytest.raises(OSError, match="socket setup failed"):
            await process.start(
                bus.client("vln", frozenset()), {"protocol": 1, "model": "x"}
            )

        assert process.returncode is not None

    asyncio.run(scenario())


def test_navigator_handshake_mismatch_reaps_process(tmp_path) -> None:
    async def scenario():
        checkpoint = tmp_path / "checkpoint"
        checkpoint.write_text("fixture")
        navigator = RPCVLNNavigator(
            (sys.executable, str(WORKER), "--wrong-model"),
            upstream_root=tmp_path,
            checkpoint=checkpoint,
            request_timeout_s=1,
        )
        bus = ToolBus()
        with pytest.raises(RPCError, match="handshake mismatch"):
            await navigator.start(
                NavTask("mismatch", NavGoal("goal", "test")),
                bus.client("vln", frozenset()),
            )
        assert navigator._process is None

    asyncio.run(scenario())


def test_late_response_is_discarded_without_killing_reader() -> None:
    async def scenario():
        bus = ToolBus()
        process = JsonLineProcess((sys.executable, str(WORKER)), request_timeout_s=1)
        await process.start(
            bus.client("vln", frozenset()), {"protocol": 1, "model": "x"}
        )
        process.request_timeout_s = 0.01
        with pytest.raises(RPCError, match="timed out"):
            await process.request("slow", {})
        process.request_timeout_s = 1
        assert await process.request("ping", {}) == "pong"
        await process.close()

    asyncio.run(scenario())


def test_model_specific_adapters_declare_distinct_requirements() -> None:
    from vln import DualVLNNavigator, JanusVLNNavigator, StreamVLNNavigator

    assert "depth" in StreamVLNNavigator.requirements["observation_channels"]
    assert JanusVLNNavigator.requirements["observation_channels"] == ["rgb"]
    dual = DualVLNNavigator(
        ("worker",),
        upstream_root="upstream",
        checkpoint="checkpoint",
    )
    assert dual.required_tools == frozenset({"nav.observe", "nav.move.discrete"})
    assert dual.requirements["motion"] == {
        "tool": "nav.move.discrete",
        "actions": ["stand_still", "forward", "turn_left", "turn_right"],
        "forward_m": 0.25,
        "turn_deg": 15.0,
    }
    assert dual.requirements["camera"] == {
        "height": 480,
        "width": 640,
        "hfov_deg": 79,
        "pitch_deg": -30,
    }
