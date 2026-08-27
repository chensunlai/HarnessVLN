from __future__ import annotations

import io
import threading
import time

import pytest

import vln.worker as worker_module
from vln.worker import WorkerRuntime


class Backend:
    model_name = "fixture"

    def load(self, hello):
        pass

    def navigate(self, instruction, options, tools, cancelled):
        return None

    def close(self):
        pass


def test_shutdown_wakes_worker_thread_blocked_on_tool_result() -> None:
    runtime = WorkerRuntime(Backend())
    runtime._writer = io.StringIO()
    runtime._jobs["job"] = {"job_id": "job", "state": "running", "reason": ""}
    errors = []

    def call_tool():
        try:
            runtime.call_tool("job", "nav.observe", {})
        except Exception as error:
            errors.append(str(error))

    thread = threading.Thread(target=call_tool)
    thread.start()
    deadline = time.monotonic() + 1
    while not runtime._tool_results and time.monotonic() < deadline:
        time.sleep(0.001)

    runtime._cancel_all()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert errors == ["worker is shutting down"]


def test_cancelled_job_keeps_worker_exclusive_until_thread_exits() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingBackend(Backend):
        def navigate(self, instruction, options, tools, cancelled):
            started.set()
            assert release.wait(timeout=1)
            return "backend returned after cancellation"

    runtime = WorkerRuntime(BlockingBackend())
    first = runtime._start_job({"instruction": "first", "options": {}})
    assert started.wait(timeout=1)

    cancelling = runtime._cancel_job({"job_id": first["job_id"]})
    assert cancelling["state"] == "cancelling"
    assert cancelling["reason"] == "cancel requested"
    with pytest.raises(RuntimeError, match="unreleased navigation job"):
        runtime._start_job({"instruction": "second", "options": {}})

    release.set()
    thread = runtime._job_threads[first["job_id"]]
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert runtime._job_status({"job_id": first["job_id"]}) == {
        "job_id": first["job_id"],
        "state": "cancelled",
        "reason": "backend returned after cancellation",
    }

    runtime._release_job({"job_id": first["job_id"]})
    second = runtime._start_job({"instruction": "second", "options": {}})
    runtime._job_threads[second["job_id"]].join(timeout=1)
    assert runtime._job_status({"job_id": second["job_id"]})["state"] == "succeeded"


def test_worker_preserves_structured_navigation_outcome() -> None:
    class StructuredBackend(Backend):
        def navigate(self, instruction, options, tools, cancelled):
            return {
                "state": "limit_reached",
                "steps": 4,
                "reason": "step limit reached: 4",
            }

    runtime = WorkerRuntime(StructuredBackend())
    started = runtime._start_job({"instruction": "local", "options": {}})
    runtime._job_threads[started["job_id"]].join(timeout=1)

    assert runtime._job_status({"job_id": started["job_id"]}) == {
        "job_id": started["job_id"],
        "state": "limit_reached",
        "steps": 4,
        "reason": "step limit reached: 4",
    }


def test_release_waits_for_reverse_calls_owned_by_backend_child_threads() -> None:
    child_threads = []
    child_errors = []
    runtime = None

    class ChildThreadBackend(Backend):
        def navigate(self, instruction, options, tools, cancelled):
            del instruction, options, cancelled

            def call_tool():
                try:
                    tools.observe()
                except Exception as error:
                    child_errors.append(str(error))

            child = threading.Thread(target=call_tool)
            child_threads.append(child)
            child.start()
            deadline = time.monotonic() + 1
            while runtime is not None and not runtime._tool_results:
                if time.monotonic() >= deadline:
                    raise TimeoutError("reverse tool call was not registered")
                time.sleep(0.001)
            return "main backend thread returned"

    runtime = WorkerRuntime(ChildThreadBackend())
    runtime._writer = io.StringIO()
    started = runtime._start_job({"instruction": "test", "options": {}})
    job_id = started["job_id"]
    runtime._job_threads[job_id].join(timeout=1)

    assert runtime._job_status({"job_id": job_id})["state"] == "succeeded"
    with pytest.raises(RuntimeError, match="pending reverse tool call"):
        runtime._release_job({"job_id": job_id})

    runtime._cancel_all()
    child_threads[0].join(timeout=1)
    assert child_errors == ["worker is shutting down"]
    assert runtime._release_job({"job_id": job_id}) == {"job_id": job_id}
    assert runtime._jobs == {}
    assert runtime._job_threads == {}
    assert runtime._cancelled == {}


def test_release_waits_until_reverse_media_decode_finishes(monkeypatch) -> None:
    decode_entered = threading.Event()
    release_decode = threading.Event()
    outputs = []

    def blocking_decode(value):
        decode_entered.set()
        assert release_decode.wait(timeout=1)
        return {"decoded": value}

    monkeypatch.setattr(worker_module, "decode_media_refs", blocking_decode)
    runtime = WorkerRuntime(Backend())
    runtime._writer = io.StringIO()
    runtime._jobs["job"] = {
        "job_id": "job",
        "state": "running",
        "reason": "",
    }
    runtime._cancelled["job"] = threading.Event()
    navigation_thread = threading.Thread(target=lambda: None)
    navigation_thread.start()
    navigation_thread.join(timeout=1)
    runtime._job_threads["job"] = navigation_thread

    caller = threading.Thread(
        target=lambda: outputs.append(runtime.call_tool("job", "nav.observe", {}))
    )
    caller.start()
    deadline = time.monotonic() + 1
    while not runtime._tool_results and time.monotonic() < deadline:
        time.sleep(0.001)
    call_id = next(iter(runtime._tool_results))
    runtime._set_job("job", "succeeded", "done")
    runtime._accept_tool_result(
        {"id": call_id, "ok": True, "result": {"path": "media"}}
    )
    assert decode_entered.wait(timeout=1)

    with pytest.raises(RuntimeError, match="pending reverse tool call"):
        runtime._release_job({"job_id": "job"})

    release_decode.set()
    caller.join(timeout=1)
    assert not caller.is_alive()
    assert outputs == [{"decoded": {"path": "media"}}]
    assert runtime._release_job({"job_id": "job"}) == {"job_id": "job"}
