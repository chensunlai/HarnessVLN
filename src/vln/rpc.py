from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from harness.errors import HarnessError
from harness.media import FileArrayStore
from harness.output import ModuleOutput, NULL_MODULE_OUTPUT
from harness.tool_bus import Tool, ToolClient
from schemas import NavTask


class RPCError(HarnessError):
    pass


class JsonLineProcess:
    """Bidirectional JSONL transport with reverse tool calls from a worker."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        request_timeout_s: float = 30.0,
    ) -> None:
        if not command:
            raise ValueError("worker command must not be empty")
        self.command = tuple(command)
        self.cwd = Path(cwd).resolve() if cwd else None
        self.env = dict(env or {})
        self.request_timeout_s = request_timeout_s
        self.stdout_tail: deque[str] = deque(maxlen=100)
        self.stderr_tail: deque[str] = deque(maxlen=100)
        self._process: asyncio.subprocess.Process | None = None
        self._tools: ToolClient | None = None
        self._pending: dict[str, asyncio.Future[Any]] = {}
        self._expired_ids: set[str] = set()
        self._expired_order: deque[str] = deque()
        self._reader_task: asyncio.Task[None] | None = None
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._protocol_reader: asyncio.StreamReader | None = None
        self._protocol_writer: asyncio.StreamWriter | None = None
        self._tool_tasks: set[asyncio.Task[None]] = set()
        self._job_tool_tasks: dict[str, set[asyncio.Task[None]]] = {}
        self._closed_jobs: set[str] = set()
        self._write_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._closing = False
        self._closed = False
        self._failure: RPCError | None = None
        self._process_group_fenced = False
        self._media = FileArrayStore()

    @property
    def returncode(self) -> int | None:
        return self._process.returncode if self._process is not None else None

    @property
    def terminated(self) -> bool:
        return self._process is None or self._process.returncode is not None

    @property
    def quiescent(self) -> bool:
        return (
            self.terminated
            and not self._tool_tasks
            and (self._process is None or self._process_group_fenced)
        )

    @property
    def active(self) -> bool:
        return (
            not self._closed
            and not self._closing
            and self._process is not None
            and self._process.returncode is None
            and self._failure is None
        )

    def bind_tools(self, tools: ToolClient) -> None:
        if not self.active:
            raise RPCError("worker is not active")
        if self._tools is not None:
            raise RPCError("worker is still bound to the previous task")
        if self._tool_tasks or self._job_tool_tasks:
            raise RPCError("worker still has reverse tool calls from the previous task")
        self._tools = tools

    def detach_tools(self) -> None:
        if self._tool_tasks or self._job_tool_tasks:
            raise RPCError("cannot detach tools while reverse calls are active")
        self._tools = None

    async def start(
        self, tools: ToolClient, hello: Mapping[str, Any]
    ) -> dict[str, Any]:
        if self._process is not None:
            raise RPCError("worker is already started")
        environment = os.environ.copy()
        environment.update(self.env)
        self._tools = tools
        parent_socket, child_socket = socket.socketpair()
        parent_socket.setblocking(False)
        child_socket.set_inheritable(True)
        environment["HARNESS_VLN_RPC_FD"] = str(child_socket.fileno())
        spawn_task = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *self.command,
                cwd=self.cwd,
                env=environment,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                pass_fds=(child_socket.fileno(),),
                start_new_session=True,
            ),
            name="vln-rpc-spawn",
        )
        try:
            self._process = await asyncio.shield(spawn_task)
            self._process_group_fenced = False
        except asyncio.CancelledError as cancellation:
            while not spawn_task.done():
                try:
                    await asyncio.shield(spawn_task)
                except asyncio.CancelledError:
                    continue
                except BaseException:
                    break
            try:
                self._process = spawn_task.result()
                self._process_group_fenced = False
            except BaseException as spawn_error:
                parent_socket.close()
                raise cancellation from spawn_error
            parent_socket.close()
            try:
                await self.close()
            except BaseException as cleanup_error:
                raise cancellation from cleanup_error
            raise cancellation
        except OSError as error:
            parent_socket.close()
            raise RPCError(
                f"failed to start worker {self.command[0]}: {error}"
            ) from error
        except BaseException:
            parent_socket.close()
            raise
        finally:
            child_socket.close()
        self._stdout_task = asyncio.create_task(
            self._read_log("stdout"), name="vln-rpc-stdout"
        )
        self._stderr_task = asyncio.create_task(
            self._read_log("stderr"), name="vln-rpc-stderr"
        )
        connected = False
        try:
            self._protocol_reader, self._protocol_writer = (
                await asyncio.open_connection(sock=parent_socket)
            )
            connected = True
            self._reader_task = asyncio.create_task(
                self._read_protocol(), name="vln-rpc-reader"
            )
            response = await self.request("hello", dict(hello))
        except BaseException as error:
            if not connected:
                parent_socket.close()
            try:
                await self.close()
            except BaseException as cleanup_error:
                if cleanup_error is error:
                    raise
                raise error from cleanup_error
            raise
        if not isinstance(response, dict):
            handshake_error = RPCError("worker hello response must be an object")
            try:
                await self.close()
            except BaseException as cleanup_error:
                raise handshake_error from cleanup_error
            raise handshake_error
        return response

    async def request(self, method: str, params: Mapping[str, Any]) -> Any:
        return await self._request(method, params)

    async def _request(
        self,
        method: str,
        params: Mapping[str, Any],
        *,
        allow_closing: bool = False,
        timeout_s: float | None = None,
    ) -> Any:
        if (
            self._closed
            or (self._closing and not allow_closing)
            or self._process is None
        ):
            raise RPCError("worker is not active")
        if self._failure is not None:
            raise self._failure
        request_id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._send(
                {
                    "type": "request",
                    "id": request_id,
                    "method": method,
                    "params": dict(params),
                }
            )
            timeout = self.request_timeout_s if timeout_s is None else timeout_s
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError as error:
            self._remember_expired(request_id)
            raise RPCError(f"worker request timed out: {method}") from error
        except asyncio.CancelledError:
            self._remember_expired(request_id)
            raise
        finally:
            self._pending.pop(request_id, None)

    async def close(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._close_impl(), name="vln-rpc-close"
            )
        elif self._close_task.done() and not self.quiescent:
            try:
                self._close_task.exception()
            except asyncio.CancelledError:
                pass
            self._close_task = asyncio.create_task(
                self._reap_impl(), name="vln-rpc-reap-retry"
            )
        close_task = self._close_task
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError as cancellation:
            # Process cleanup is bounded and must outlive cancellation of its caller.
            try:
                await asyncio.shield(close_task)
            except BaseException as cleanup_error:
                if not self.quiescent:
                    try:
                        await self._retry_reap()
                    except BaseException as reap_error:
                        raise cancellation from reap_error
                raise cancellation from cleanup_error
            raise cancellation
        except BaseException as close_error:
            if not self.quiescent:
                try:
                    await self._retry_reap()
                except BaseException as reap_error:
                    raise close_error from reap_error
            raise

    async def _retry_reap(self) -> None:
        self._close_task = asyncio.create_task(
            self._reap_impl(), name="vln-rpc-reap-retry"
        )
        await asyncio.shield(self._close_task)

    async def _close_impl(self) -> None:
        if self._closed:
            return
        self._closing = True
        self._tools = None
        process = self._process
        shutdown_acknowledged = False
        shutdown_failure_code: int | None = None
        try:
            if (
                process is not None
                and process.returncode is None
                and self._protocol_writer is not None
            ):
                try:
                    await self._request(
                        "shutdown", {}, allow_closing=True, timeout_s=0.5
                    )
                    shutdown_acknowledged = True
                except (RPCError, asyncio.TimeoutError):
                    pass
            if (
                shutdown_acknowledged
                and process is not None
                and process.returncode is None
            ):
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
            if process is not None and process.returncode is None:
                self._signal_process_group(process, signal.SIGTERM)
                try:
                    await asyncio.wait_for(process.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    self._signal_process_group(process, signal.SIGKILL)
                    await asyncio.wait_for(process.wait(), timeout=1.0)
            if (
                shutdown_acknowledged
                and process is not None
                and process.returncode not in (None, 0)
            ):
                shutdown_failure_code = process.returncode
        finally:
            # This synchronous fence must run even when the reaper task itself is
            # cancelled during graceful shutdown.
            self._closed = True
            tool_tasks = tuple(self._tool_tasks)
            for tool_task in tool_tasks:
                tool_task.cancel()
            if process is not None:
                self._fence_process_group(process)
            if self._protocol_writer is not None:
                self._protocol_writer.close()
            self._media.close()
            self._job_tool_tasks.clear()
            try:
                if process is not None and process.returncode is None:
                    await asyncio.wait_for(process.wait(), timeout=1.0)
                if tool_tasks:
                    _, pending_tools = await asyncio.wait(tool_tasks, timeout=0.5)
                    if pending_tools:
                        self._mark_failure(
                            RPCError(
                                f"{len(pending_tools)} reverse tool call(s) ignored "
                                "cancellation during close"
                            )
                        )
                try:
                    if self._protocol_writer is not None:
                        await asyncio.wait_for(
                            self._protocol_writer.wait_closed(), timeout=0.5
                        )
                except (
                    asyncio.TimeoutError,
                    BrokenPipeError,
                    ConnectionResetError,
                    OSError,
                ):
                    pass
                tasks = [
                    stream_task
                    for stream_task in (
                        self._reader_task,
                        self._stdout_task,
                        self._stderr_task,
                    )
                    if stream_task is not None
                ]
                if tasks:
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*tasks, return_exceptions=True), timeout=1.0
                        )
                    except asyncio.TimeoutError:
                        for stream_task in tasks:
                            if not stream_task.done():
                                stream_task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                self._mark_failure(RPCError("worker closed"))
                self._media.close()
                self._job_tool_tasks.clear()
        if shutdown_failure_code is not None:
            raise RPCError(
                f"worker shutdown failed with code {shutdown_failure_code}; "
                f"stderr: {self._stderr_summary()}"
            )

    async def _reap_impl(self) -> None:
        """Retry the non-graceful part of cleanup after a failed close task."""
        self._closing = True
        self._closed = True
        self._tools = None
        tool_tasks = tuple(self._tool_tasks)
        stream_tasks = tuple(
            task
            for task in (self._reader_task, self._stdout_task, self._stderr_task)
            if task is not None
        )
        for task in (*tool_tasks, *stream_tasks):
            task.cancel()
        if self._protocol_writer is not None:
            self._protocol_writer.close()
        self._media.close()
        self._job_tool_tasks.clear()
        process = self._process
        if process is not None:
            self._fence_process_group(process)
            if process.returncode is None:
                await asyncio.wait_for(process.wait(), timeout=1.0)
        if tool_tasks or stream_tasks:
            await asyncio.wait((*tool_tasks, *stream_tasks), timeout=0.5)
        self._mark_failure(RPCError("worker closed"))

    def seal_job(self, job_id: str) -> None:
        self._closed_jobs.add(job_id)

    async def close_job(self, job_id: str) -> None:
        self.seal_job(job_id)
        tasks = tuple(self._job_tool_tasks.get(job_id, ()))
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            errors = [result for result in results if isinstance(result, BaseException)]
            if errors:
                error = RPCError(
                    f"reverse tool call failed while closing {job_id}: "
                    f"{type(errors[0]).__name__}: {errors[0]}"
                )
                self._mark_failure(error)
                raise error
        await asyncio.sleep(0)
        if self._failure is not None:
            raise self._failure
        self._media.release(job_id)

    async def _read_protocol(self) -> None:
        assert self._process is not None and self._protocol_reader is not None
        try:
            while line := await self._protocol_reader.readline():
                try:
                    message = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RPCError("worker sent invalid JSONL protocol data") from error
                await self._dispatch(message)
            returncode = await self._process.wait()
            raise RPCError(
                f"worker exited with code {returncode}; stderr: {self._stderr_summary()}"
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._mark_failure(
                error if isinstance(error, RPCError) else RPCError(str(error))
            )

    async def _dispatch(self, message: Any) -> None:
        if not isinstance(message, dict):
            raise RPCError("worker message must be an object")
        message_type = message.get("type")
        if message_type == "response":
            request_id = message.get("id")
            if not isinstance(request_id, str):
                raise RPCError("worker response id must be a string")
            if request_id in self._expired_ids:
                self._expired_ids.discard(request_id)
                return
            future = self._pending.get(request_id)
            if future is None or future.done():
                raise RPCError(f"response for unknown request: {request_id}")
            if message.get("ok") is True:
                future.set_result(message.get("result"))
            else:
                future.set_exception(
                    RPCError(str(message.get("error", "worker error")))
                )
            return
        if message_type == "tool_call":
            job_id = message.get("job_id")
            if not isinstance(job_id, str):
                raise RPCError("worker tool_call job_id must be a string")
            if job_id in self._closed_jobs:
                raise RPCError(f"tool_call arrived after job closed: {job_id}")
            tools = self._tools
            if tools is None:
                raise RPCError(f"tool_call arrived without an active task: {job_id}")
            task = asyncio.create_task(
                self._handle_tool_call(message, tools), name="vln-tool-call"
            )
            self._tool_tasks.add(task)
            self._job_tool_tasks.setdefault(job_id, set()).add(task)

            def discard(done: asyncio.Task[None]) -> None:
                self._tool_tasks.discard(done)
                job_tasks = self._job_tool_tasks.get(job_id)
                if job_tasks is not None:
                    job_tasks.discard(done)
                    if not job_tasks:
                        self._job_tool_tasks.pop(job_id, None)
                if not done.cancelled():
                    error = done.exception()
                    if error is not None:
                        self._mark_failure(
                            error
                            if isinstance(error, RPCError)
                            else RPCError(
                                f"reverse tool response failed: "
                                f"{type(error).__name__}: {error}"
                            )
                        )

            task.add_done_callback(discard)
            return
        raise RPCError(f"unknown worker message type: {message_type!r}")

    async def _handle_tool_call(
        self, message: dict[str, Any], tools: ToolClient
    ) -> None:
        call_id = message.get("id")
        job_id = message.get("job_id")
        name = message.get("name")
        arguments = message.get("arguments", {})
        if (
            not isinstance(call_id, str)
            or not isinstance(job_id, str)
            or not isinstance(name, str)
            or not isinstance(arguments, dict)
        ):
            raise RPCError("malformed worker tool_call")
        try:
            result = await tools.call(name, arguments)
            response = {
                "type": "tool_result",
                "id": call_id,
                "ok": True,
                "result": result,
            }
        except Exception as error:
            response = {
                "type": "tool_result",
                "id": call_id,
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
        await self._send(response, media_scope=job_id)

    async def _send(
        self, message: Mapping[str, Any], *, media_scope: str | None = None
    ) -> None:
        if self._closed:
            raise RPCError("worker is closed")
        assert self._process is not None and self._protocol_writer is not None
        try:
            payload = (
                json.dumps(
                    self._media.encode(message, scope=media_scope),
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
        except (TypeError, ValueError) as error:
            raise RPCError(
                f"protocol value is not JSON serializable: {error}"
            ) from error
        async with self._write_lock:
            try:
                self._protocol_writer.write(payload)
                await self._protocol_writer.drain()
            except (BrokenPipeError, ConnectionResetError, RuntimeError) as error:
                failure = RPCError(
                    f"worker pipe closed; stderr: {self._stderr_summary()}"
                )
                self._mark_failure(failure)
                raise failure from error

    async def _read_log(self, stream_name: str) -> None:
        assert self._process is not None
        stream = getattr(self._process, stream_name)
        assert stream is not None
        target = self.stdout_tail if stream_name == "stdout" else self.stderr_tail
        while line := await stream.readline():
            target.append(line.decode("utf-8", errors="replace").rstrip())

    def _stderr_summary(self) -> str:
        return " | ".join(self.stderr_tail) or "<empty>"

    def _mark_failure(self, error: RPCError) -> None:
        if self._failure is None:
            self._failure = error
        for future in tuple(self._pending.values()):
            if not future.done():
                future.set_exception(self._failure)

    def _remember_expired(self, request_id: str) -> None:
        if len(self._expired_order) >= 1024:
            oldest = self._expired_order.popleft()
            self._expired_ids.discard(oldest)
        self._expired_order.append(request_id)
        self._expired_ids.add(request_id)

    @staticmethod
    def _signal_process_group(
        process: asyncio.subprocess.Process, process_signal: signal.Signals
    ) -> None:
        try:
            os.killpg(process.pid, process_signal)
        except ProcessLookupError:
            pass

    def _fence_process_group(self, process: asyncio.subprocess.Process) -> None:
        self._signal_process_group(process, signal.SIGKILL)
        self._process_group_fenced = True


class RPCVLNNavigator:
    protocol_version = 2
    model_name = "rpc"
    required_tools: frozenset[str] = frozenset()
    requirements: dict[str, Any] = {}
    _active_states = frozenset({"running", "cancelling"})
    _terminal_states = frozenset({"succeeded", "cancelled", "failed"})

    def __init__(
        self,
        command: Sequence[str],
        *,
        upstream_root: str | Path,
        checkpoint: str | Path,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        worker_options: Mapping[str, Any] | None = None,
        request_timeout_s: float = 300.0,
        local_max_steps: int = 16,
        status_poll_s: float = 0.1,
    ) -> None:
        if local_max_steps <= 0:
            raise ValueError("local_max_steps must be positive")
        if status_poll_s < 0:
            raise ValueError("status_poll_s must not be negative")
        self.command = tuple(command)
        self.upstream_root = Path(upstream_root)
        self.checkpoint = Path(checkpoint)
        self.cwd = Path(cwd) if cwd else self.upstream_root
        self.env = dict(env or {})
        self.worker_options = dict(worker_options or {})
        self.request_timeout_s = request_timeout_s
        self.local_max_steps = local_max_steps
        self.status_poll_s = status_poll_s
        self._task: NavTask | None = None
        self._process: JsonLineProcess | None = None
        self._retired_process: JsonLineProcess | None = None
        self._active_jobs: set[str] = set()
        self._terminal_jobs: dict[str, dict[str, Any]] = {}
        self._session_scoped = False
        self._lifecycle_lock = asyncio.Lock()

    def enable_session_scope(self) -> None:
        if self._process is not None or self._retired_process is not None:
            raise HarnessError(
                "session scope must be enabled before starting the VLN worker"
            )
        self._session_scoped = True

    async def start(
        self,
        task: NavTask,
        tools: ToolClient,
        output: ModuleOutput = NULL_MODULE_OUTPUT,
    ):
        async with self._lifecycle_lock:
            if self._task is not None:
                raise HarnessError(
                    f"{self.model_name} navigator already has an active task"
                )
            self._validate_resources()
            if self._retired_process is not None:
                await self._close_process_locked()
                if self._retired_process is not None:
                    raise RPCError("previous VLN worker has not terminated")
            if self._process is not None and not self._process.active:
                await self._close_process_locked()
            if self._process is None:
                process = JsonLineProcess(
                    self.command,
                    cwd=self.cwd,
                    env=self.env,
                    request_timeout_s=self.request_timeout_s,
                )
                self._process = process
                try:
                    hello = await process.start(
                        tools,
                        {
                            "protocol": self.protocol_version,
                            "model": self.model_name,
                            "upstream_root": str(self.upstream_root.resolve()),
                            "checkpoint": str(self.checkpoint.resolve()),
                            "options": self.worker_options,
                        },
                    )
                    capabilities = hello.get("capabilities")
                    if (
                        hello.get("protocol") != self.protocol_version
                        or hello.get("model") != self.model_name
                        or not isinstance(capabilities, list)
                        or "navigate.release" not in capabilities
                    ):
                        raise RPCError(f"worker handshake mismatch: {hello!r}")
                except BaseException as error:
                    await self._close_after_error_locked(error)
                    raise
            else:
                try:
                    self._process.bind_tools(tools)
                except BaseException as error:
                    await self._close_after_error_locked(error)
                    raise
            self._terminal_jobs.clear()
            self._task = task
            output.record(
                {
                    "navigator": type(self).__name__,
                    "model": self.model_name,
                    "protocol": self.protocol_version,
                    "requirements": self.requirements,
                    "local_max_steps": self.local_max_steps,
                }
            )
        return (
            Tool(
                "vln.navigate.task",
                "Run the complete task instruction with the VLN model and block "
                "until it finishes. This compatibility tool is intended for "
                "passthrough agents.",
                {
                    "type": "object",
                    "properties": {
                        "instruction": {"type": "string", "minLength": 1},
                    },
                    "required": ["instruction"],
                    "additionalProperties": False,
                },
                self._navigate_task,
                writes=True,
            ),
            Tool(
                "vln.navigate.local",
                "Navigate to one landmark, opening, object, or free-space region "
                "visible in the latest observation, then block until the local "
                "attempt finishes. The instruction must not contain the complete "
                "task or unseen waypoints.",
                {
                    "type": "object",
                    "properties": {
                        "instruction": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 256,
                            "description": (
                                "One short instruction grounded in a target visible "
                                "in the latest observation."
                            ),
                        },
                    },
                    "required": ["instruction"],
                    "additionalProperties": False,
                },
                self._navigate_local,
                writes=True,
            ),
        )

    async def _navigate_task(self, actor: str, arguments: dict[str, Any]) -> Any:
        return await self._run_blocking_job(actor, arguments["instruction"], {})

    async def _navigate_local(self, actor: str, arguments: dict[str, Any]) -> Any:
        return await self._run_blocking_job(
            actor,
            arguments["instruction"],
            {"max_steps": self.local_max_steps},
        )

    async def _run_blocking_job(
        self, actor: str, instruction: str, options: Mapping[str, Any]
    ) -> dict[str, Any]:
        started = await self._start_job(
            actor,
            {"instruction": instruction, "options": dict(options)},
        )
        job_id = started["job_id"]
        try:
            while True:
                status = await self._status_job(actor, {"job_id": job_id})
                if status["state"] in self._terminal_states:
                    return {
                        key: value for key, value in status.items() if key != "job_id"
                    }
                await asyncio.sleep(self.status_poll_s)
        except asyncio.CancelledError as cancellation:
            try:
                await asyncio.shield(self._cancel_if_active(job_id))
            except BaseException as cleanup_error:
                raise cancellation from cleanup_error
            raise

    async def _cancel_if_active(self, job_id: str) -> None:
        async with self._lifecycle_lock:
            if job_id not in self._active_jobs:
                return
            try:
                await self._cancel_job_locked(job_id)
            except BaseException as error:
                await self._close_after_error_locked(error)
                raise

    async def _start_job(self, actor: str, arguments: dict[str, Any]) -> Any:
        del actor
        async with self._lifecycle_lock:
            if self._process is None or self._task is None:
                raise RPCError("navigator has no active task")
            if self._active_jobs:
                raise RPCError("navigator already has an active navigation job")
            process = self._process
            try:
                result = await process.request(
                    "navigate.start",
                    {
                        "task_id": self._task.task_id,
                        "instruction": arguments["instruction"],
                        "options": arguments["options"],
                    },
                )
                if (
                    not isinstance(result, Mapping)
                    or not isinstance(result.get("job_id"), str)
                    or not result["job_id"]
                ):
                    raise RPCError("worker navigate.start returned an invalid job id")
            except BaseException as error:
                # The worker may have started a job even when its reply was lost.
                await self._close_after_error_locked(error)
                raise
            job_id = result["job_id"]
            self._active_jobs.add(job_id)
            return dict(result)

    async def _status_job(self, actor: str, arguments: dict[str, Any]) -> Any:
        del actor
        job_id = arguments["job_id"]
        async with self._lifecycle_lock:
            if job_id in self._terminal_jobs:
                return dict(self._terminal_jobs[job_id])
            self._require_active_job_locked(job_id)
            try:
                assert self._process is not None
                result = self._validate_status(
                    await self._process.request("navigate.status", arguments), job_id
                )
                if result["state"] in self._terminal_states:
                    await self._release_job_locked(job_id)
                    self._terminal_jobs[job_id] = result
                return dict(result)
            except BaseException as error:
                await self._close_after_error_locked(error)
                raise

    async def _cancel_job(self, actor: str, arguments: dict[str, Any]) -> Any:
        del actor
        job_id = arguments["job_id"]
        async with self._lifecycle_lock:
            if job_id in self._terminal_jobs:
                return dict(self._terminal_jobs[job_id])
            self._require_active_job_locked(job_id)
            try:
                return await self._cancel_job_locked(job_id)
            except BaseException as error:
                await self._close_after_error_locked(error)
                raise

    async def stop(self, reason: str) -> None:
        del reason
        async with self._lifecycle_lock:
            try:
                for job_id in tuple(self._active_jobs):
                    await self._cancel_job_locked(job_id)
                if (
                    self._process is not None
                    and self._process.active
                    and self._session_scoped
                ):
                    self._process.detach_tools()
                    self._terminal_jobs.clear()
                    self._task = None
                else:
                    await self._close_process_locked()
            except BaseException as error:
                await self._close_after_error_locked(error)
                raise

    async def close_session(self) -> None:
        async with self._lifecycle_lock:
            await self._close_process_locked()

    async def _cancel_job_locked(self, job_id: str) -> dict[str, Any]:
        if job_id in self._terminal_jobs:
            return dict(self._terminal_jobs[job_id])
        self._require_active_job_locked(job_id)
        assert self._process is not None
        result = self._validate_status(
            await self._process.request("navigate.cancel", {"job_id": job_id}),
            job_id,
        )
        while result["state"] in self._active_states:
            await asyncio.sleep(0.05)
            result = self._validate_status(
                await self._process.request("navigate.status", {"job_id": job_id}),
                job_id,
            )
        await self._release_job_locked(job_id)
        self._terminal_jobs[job_id] = result
        return dict(result)

    async def _release_job_locked(self, job_id: str) -> None:
        if job_id not in self._active_jobs:
            return
        assert self._process is not None
        process = self._process
        process.seal_job(job_id)
        released = await process.request("navigate.release", {"job_id": job_id})
        if not isinstance(released, Mapping) or released.get("job_id") != job_id:
            raise RPCError("worker navigate.release returned an invalid job id")
        await process.close_job(job_id)
        if not process.active:
            raise RPCError("worker became unhealthy while releasing a job")
        self._active_jobs.remove(job_id)

    async def _close_process_locked(self) -> None:
        process = self._process or self._retired_process
        self._process = None
        self._retired_process = process
        self._active_jobs.clear()
        self._terminal_jobs.clear()
        self._task = None
        if process is not None:
            try:
                await process.close()
            finally:
                if process.quiescent and self._retired_process is process:
                    self._retired_process = None
            if self._retired_process is process:
                raise RPCError(
                    "previous VLN worker still has reverse tool calls in quarantine"
                )

    async def _close_after_error_locked(self, original: BaseException) -> None:
        try:
            await self._close_process_locked()
        except BaseException as cleanup_error:
            if cleanup_error is original:
                raise
            raise original from cleanup_error

    def _require_active_job_locked(self, job_id: str) -> None:
        if job_id not in self._active_jobs or self._process is None:
            raise RPCError(f"unknown navigation job: {job_id}")

    def _validate_status(self, value: Any, job_id: str) -> dict[str, Any]:
        if (
            not isinstance(value, Mapping)
            or value.get("job_id") != job_id
            or value.get("state") not in self._active_states | self._terminal_states
            or not isinstance(value.get("reason"), str)
        ):
            raise RPCError(f"worker returned an invalid status for {job_id}: {value!r}")
        return dict(value)

    def _validate_resources(self) -> None:
        if not self.upstream_root.is_dir():
            raise HarnessError(
                f"{self.model_name} upstream root not found: {self.upstream_root}"
            )
        if not self.checkpoint.exists():
            raise HarnessError(
                f"{self.model_name} checkpoint not found: {self.checkpoint}"
            )
