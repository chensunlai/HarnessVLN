from __future__ import annotations

import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from domain.contracts import DomainResult, DomainSpec, NavigationEpisode, Terminal
from domain.errors import HarnessError
from domain.io import write_json, write_jsonl
from domain.modules import Module, ModuleRegistry, validate_metrics
from domain.register import DomainRegister
from domain.workspace import Workspace


@dataclass(slots=True)
class _ModuleThread:
    name: str
    module: Module
    thread: threading.Thread | None = None
    error: BaseException | None = None

    def start(self) -> None:
        self.thread = threading.Thread(
            target=self._run,
            name=f"domain-{self.name}",
            daemon=True,
        )
        self.thread.start()

    def _run(self) -> None:
        try:
            self.module.run()
        except BaseException as error:
            self.error = error
            self.module._fail(error)

    @property
    def alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def join(self, timeout_s: float) -> None:
        if self.thread is not None:
            self.thread.join(timeout_s)


class DomainRuntime:
    """Run one episode. It supervises modules but never drives navigation."""

    def run(
        self,
        episode: NavigationEpisode,
        spec: DomainSpec,
        output_dir: str | Path,
        *,
        domain_id: str | None = None,
    ) -> DomainResult:
        identifier = _slug(domain_id or uuid.uuid4().hex)
        root = Path(output_dir).expanduser().resolve() / identifier
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / "episode.json", episode.as_dict())
        write_json(root / "domain_config.json", spec.as_dict())

        cancelled = threading.Event()
        register = DomainRegister()
        workspace = Workspace(root / "workspace", spec.workspace)
        workspace.mount(register)
        register.register_reference("domain", "domain.id", lambda: identifier)
        register.register_reference(
            "domain",
            "domain.episode",
            lambda: episode.as_dict(include_truth=False),
            description="Public navigation episode data.",
        )
        register.register_reference("domain", "domain.cancelled", cancelled.is_set)

        modules = ModuleRegistry(spec.all_modules)
        errors: list[str] = []
        threads: dict[str, _ModuleThread] = {}
        terminal: Terminal | None = None
        environment_result: dict[str, Any] = {}
        metrics: dict[str, float] = {}

        try:
            modules.build(
                domain_id=identifier,
                episode=episode,
                register=register,
                workspace=workspace,
                cancelled=cancelled,
            )
            modules.mount()
            missing = {"env.step", "env.stop"} - register.names
            if missing:
                raise HarnessError(
                    f"environment did not register required functions: {', '.join(sorted(missing))}"
                )
            write_json(root / "register.json", register.manifest())

            env_thread = _ModuleThread("env", modules.environment)
            threads["env"] = env_thread
            env_thread.start()
            startup_deadline = time.monotonic() + min(spec.timeout_s, spec.shutdown_timeout_s)
            while not modules.environment.wait_ready(0.02):
                if env_thread.error is not None or time.monotonic() >= startup_deadline:
                    break
            if env_thread.error is not None:
                raise env_thread.error
            if not modules.environment.wait_ready(0):
                raise HarnessError("environment startup timed out")

            for name, module in modules.modules.items():
                if name == "env":
                    continue
                thread = _ModuleThread(name, module)
                threads[name] = thread
                thread.start()

            deadline = time.monotonic() + spec.timeout_s
            while terminal is None:
                terminal = modules.environment.wait_terminal(0.02)
                failed = next(
                    ((name, item.error) for name, item in threads.items() if item.error),
                    None,
                )
                if failed is not None and terminal is None:
                    name, error = failed
                    terminal = self._request_stop(
                        register,
                        "failed",
                        f"module {name} failed: {type(error).__name__}: {error}",
                        name,
                    )
                if time.monotonic() >= deadline and terminal is None:
                    terminal = self._request_stop(
                        register, "timeout", "Domain execution timed out", "domain"
                    )
        except BaseException as error:
            errors.append(f"runtime: {type(error).__name__}: {error}")
            terminal = terminal or self._request_stop(
                register,
                "failed",
                f"Domain startup failed: {type(error).__name__}: {error}",
                "domain",
            )
        finally:
            register.close_writes()
            cancelled.set()
            for thread in threads.values():
                thread.join(spec.shutdown_timeout_s)
                if thread.alive:
                    errors.append(f"module {thread.name} did not stop before shutdown timeout")
                elif thread.error is not None:
                    message = f"module {thread.name}: {type(thread.error).__name__}: {thread.error}"
                    if message not in errors:
                        errors.append(message)

            if terminal is None:
                terminal = Terminal("failed", "environment produced no terminal state", "domain")
            if "env" in modules.modules:
                try:
                    environment_result = dict(modules.environment.result())
                except BaseException as error:
                    errors.append(f"environment result: {type(error).__name__}: {error}")
            if "metric" in modules.modules:
                try:
                    metrics = validate_metrics(
                        modules.metric.evaluate(terminal, environment_result)
                    )
                except BaseException as error:
                    errors.append(f"metric evaluate: {type(error).__name__}: {error}")
            if modules.modules:
                errors.extend(modules.close())

        result = DomainResult(
            identifier,
            episode.episode_id,
            terminal,
            environment_result,
            metrics,
            modules.manifest() if modules.modules else {},
            tuple(errors),
            str(workspace.root),
        )
        write_jsonl(root / "calls.jsonl", register.records)
        write_json(root / "result.json", result)
        return result

    @staticmethod
    def _request_stop(
        register: DomainRegister,
        status: str,
        reason: str,
        actor: str,
    ) -> Terminal:
        try:
            value = register.call(
                "domain", "env.stop", {"status": status, "reason": reason, "actor": actor}
            )
            return Terminal(
                str(value.get("status", status)),
                str(value.get("reason", reason)),
                str(value.get("actor", actor)),
            )
        except BaseException as error:
            return Terminal(
                "failed",
                f"{reason}; env.stop failed: {type(error).__name__}: {error}",
                "domain",
            )


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not result:
        raise HarnessError(f"invalid Domain identifier: {value!r}")
    return result
