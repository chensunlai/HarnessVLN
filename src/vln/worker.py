from __future__ import annotations

import json
import os
import socket
import threading
import traceback
import uuid
from collections.abc import Mapping
from typing import Any, Protocol

from harness.media import decode_media_refs


class WorkerBackend(Protocol):
    model_name: str

    def load(self, hello: Mapping[str, Any]) -> None: ...

    def navigate(
        self,
        instruction: str,
        options: Mapping[str, Any],
        tools: WorkerTools,
        cancelled: threading.Event,
    ) -> str | Mapping[str, Any] | None: ...

    def close(self) -> None: ...


class WorkerTools:
    def __init__(self, runtime: WorkerRuntime, job_id: str) -> None:
        self._runtime = runtime
        self.job_id = job_id

    def call(self, name: str, arguments: Mapping[str, Any] | None = None) -> Any:
        return self._runtime.call_tool(self.job_id, name, arguments or {})

    def observe(self) -> dict[str, Any]:
        return self.call("nav.observe")

    def move_discrete(self, action: str) -> dict[str, Any]:
        return self.call("nav.move.discrete", {"action": action})


class WorkerRuntime:
    protocol_version = 2

    def __init__(self, backend: WorkerBackend) -> None:
        self.backend = backend
        self._writer: Any = None
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._tool_results: dict[
            str, tuple[str, threading.Event, dict[str, Any] | None]
        ] = {}
        self._jobs: dict[str, dict[str, Any]] = {}
        self._job_threads: dict[str, threading.Thread] = {}
        self._cancelled: dict[str, threading.Event] = {}
        self._loaded = False

    def run(self) -> None:
        protocol_socket = socket.socket(fileno=int(os.environ["HARNESS_VLN_RPC_FD"]))
        reader = protocol_socket.makefile("r", encoding="utf-8")
        self._writer = protocol_socket.makefile("w", encoding="utf-8", buffering=1)
        try:
            for line in reader:
                message = json.loads(line)
                if message.get("type") == "tool_result":
                    self._accept_tool_result(message)
                    continue
                self._handle_request(message)
                if message.get("method") == "shutdown":
                    break
        finally:
            self._cancel_all()
            if self._loaded:
                self.backend.close()
            reader.close()
            self._writer.close()
            protocol_socket.close()

    def call_tool(self, job_id: str, name: str, arguments: Mapping[str, Any]) -> Any:
        call_id = uuid.uuid4().hex
        ready = threading.Event()
        with self._state_lock:
            job = self._jobs.get(job_id)
            if job is None or job["state"] not in {"running", "cancelling"}:
                raise RuntimeError(f"navigation job is not active: {job_id}")
            self._tool_results[call_id] = (job_id, ready, None)
        try:
            self._send(
                {
                    "type": "tool_call",
                    "id": call_id,
                    "job_id": job_id,
                    "name": name,
                    "arguments": dict(arguments),
                }
            )
            ready.wait()
            with self._state_lock:
                _, _, result = self._tool_results[call_id]
            assert result is not None
            if result.get("ok") is not True:
                raise RuntimeError(str(result.get("error", "tool call failed")))
            return decode_media_refs(result.get("result"))
        finally:
            with self._state_lock:
                self._tool_results.pop(call_id, None)

    def _handle_request(self, message: Mapping[str, Any]) -> None:
        request_id = message.get("id")
        method = message.get("method")
        params = message.get("params")
        if (
            not isinstance(request_id, str)
            or not isinstance(method, str)
            or not isinstance(params, Mapping)
        ):
            return
        try:
            if method == "hello":
                result = self._hello(params)
            elif method == "navigate.start":
                result = self._start_job(params)
            elif method == "navigate.status":
                result = self._job_status(params)
            elif method == "navigate.cancel":
                result = self._cancel_job(params)
            elif method == "navigate.release":
                result = self._release_job(params)
            elif method == "shutdown":
                self._cancel_all()
                result = {}
            else:
                raise ValueError(f"unknown method: {method}")
        except Exception as error:
            self._respond(request_id, False, error=f"{type(error).__name__}: {error}")
        else:
            self._respond(request_id, True, result=result)

    def _hello(self, params: Mapping[str, Any]) -> dict[str, Any]:
        if params.get("protocol") != self.protocol_version:
            raise ValueError("protocol mismatch")
        if params.get("model") != self.backend.model_name:
            raise ValueError("model mismatch")
        self.backend.load(params)
        self._loaded = True
        return {
            "protocol": self.protocol_version,
            "model": self.backend.model_name,
            "capabilities": ["navigate.release"],
        }

    def _start_job(self, params: Mapping[str, Any]) -> dict[str, str]:
        instruction = params.get("instruction")
        options = params.get("options", {})
        if not isinstance(instruction, str) or not instruction:
            raise ValueError("instruction must be non-empty")
        if not isinstance(options, Mapping):
            raise ValueError("options must be an object")
        with self._state_lock:
            if self._jobs:
                raise RuntimeError("worker already has an unreleased navigation job")
            job_id = uuid.uuid4().hex
            self._jobs[job_id] = {"job_id": job_id, "state": "running", "reason": ""}
            cancelled = threading.Event()
            self._cancelled[job_id] = cancelled
        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, instruction, dict(options), cancelled),
            name=f"vln-job-{job_id[:8]}",
            daemon=True,
        )
        self._job_threads[job_id] = thread
        thread.start()
        return {"job_id": job_id}

    def _run_job(
        self,
        job_id: str,
        instruction: str,
        options: Mapping[str, Any],
        cancelled: threading.Event,
    ) -> None:
        try:
            outcome = self.backend.navigate(
                instruction, options, WorkerTools(self, job_id), cancelled
            )
            state, reason, details = _normalize_outcome(outcome, cancelled)
            self._set_job(job_id, state, reason, **details)
        except Exception as error:
            self._set_job(
                job_id,
                "failed",
                f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(limit=20),
            )

    def _job_status(self, params: Mapping[str, Any]) -> dict[str, Any]:
        job_id = params.get("job_id")
        with self._state_lock:
            if not isinstance(job_id, str) or job_id not in self._jobs:
                raise KeyError(f"unknown job: {job_id}")
            return dict(self._jobs[job_id])

    def _cancel_job(self, params: Mapping[str, Any]) -> dict[str, Any]:
        job_id = params.get("job_id")
        with self._state_lock:
            if not isinstance(job_id, str) or job_id not in self._jobs:
                raise KeyError(f"unknown job: {job_id}")
            self._cancelled[job_id].set()
            if self._jobs[job_id]["state"] == "running":
                self._jobs[job_id].update(state="cancelling", reason="cancel requested")
            return dict(self._jobs[job_id])

    def _release_job(self, params: Mapping[str, Any]) -> dict[str, str]:
        job_id = params.get("job_id")
        with self._state_lock:
            if not isinstance(job_id, str) or job_id not in self._jobs:
                raise KeyError(f"unknown job: {job_id}")
            thread = self._job_threads[job_id]
            state = self._jobs[job_id]["state"]
        if state in {"running", "cancelling"}:
            raise RuntimeError(f"cannot release active job: {job_id}")
        thread.join(timeout=1.0)
        if thread.is_alive():
            raise RuntimeError(f"job thread did not exit: {job_id}")
        with self._state_lock:
            pending = [
                call_id
                for call_id, (owner, _, _) in self._tool_results.items()
                if owner == job_id
            ]
            if pending:
                raise RuntimeError(
                    f"job still has {len(pending)} pending reverse tool call(s): {job_id}"
                )
            self._jobs.pop(job_id)
            self._job_threads.pop(job_id)
            self._cancelled.pop(job_id)
        return {"job_id": job_id}

    def _cancel_all(self) -> None:
        with self._state_lock:
            cancellations = tuple(self._cancelled.values())
            pending = tuple(self._tool_results.items())
            for call_id, (job_id, ready, _) in pending:
                self._tool_results[call_id] = (
                    job_id,
                    ready,
                    {"ok": False, "error": "worker is shutting down"},
                )
        for cancelled in cancellations:
            cancelled.set()
        for _, (_, ready, _) in pending:
            ready.set()
        for thread in tuple(self._job_threads.values()):
            thread.join(timeout=2.0)

    def _accept_tool_result(self, message: Mapping[str, Any]) -> None:
        call_id = message.get("id")
        if not isinstance(call_id, str):
            return
        with self._state_lock:
            pending = self._tool_results.get(call_id)
            if pending is None:
                return
            job_id, ready, _ = pending
            self._tool_results[call_id] = (job_id, ready, dict(message))
        ready.set()

    def _set_job(self, job_id: str, state: str, reason: str, **extra: Any) -> None:
        with self._state_lock:
            self._jobs[job_id].update(state=state, reason=reason, **extra)

    def _respond(
        self,
        request_id: str,
        ok: bool,
        *,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        message = {"type": "response", "id": request_id, "ok": ok}
        if ok:
            message["result"] = result
        else:
            message["error"] = error
        self._send(message)

    def _send(self, message: Mapping[str, Any]) -> None:
        with self._write_lock:
            self._writer.write(json.dumps(message, separators=(",", ":")) + "\n")
            self._writer.flush()


def _normalize_outcome(
    value: str | Mapping[str, Any] | None,
    cancelled: threading.Event,
) -> tuple[str, str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        state = "cancelled" if cancelled.is_set() else "succeeded"
        return state, value or state, {}

    state = value.get("state")
    reason = value.get("reason")
    if state not in {"succeeded", "limit_reached", "cancelled", "failed"}:
        raise ValueError(f"backend returned invalid navigation state: {state!r}")
    if not isinstance(reason, str):
        raise ValueError("backend navigation reason must be a string")
    details = {
        str(key): item
        for key, item in value.items()
        if key not in {"state", "reason"}
    }
    steps = details.get("steps")
    if steps is not None and (type(steps) is not int or steps < 0):
        raise ValueError("backend navigation steps must be a non-negative integer")
    if cancelled.is_set() and state != "failed":
        state = "cancelled"
    return state, reason, details


def run_worker(backend: WorkerBackend) -> None:
    WorkerRuntime(backend).run()
