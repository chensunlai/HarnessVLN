from __future__ import annotations

import threading
import time

import pytest

from domain.errors import DomainClosedError, RegisterError
from domain.register import DomainRegister


def test_functions_references_and_openai_schema() -> None:
    register = DomainRegister()
    value = {"count": 0}
    register.register_function(
        "counter",
        "counter.add",
        lambda amount: value.__setitem__("count", value["count"] + amount) or value["count"],
        description="Add to the counter.",
        parameters={
            "type": "object",
            "properties": {"amount": {"type": "integer"}},
            "required": ["amount"],
        },
        mutates=True,
        serial_key="counter",
    )
    register.register_reference("counter", "counter.value", lambda: value["count"])

    assert register.call("test", "counter.add", {"amount": 2}) == 2
    assert register.read("test", "counter.value") == 2
    tools, names = register.openai_toolset()
    assert tools[0]["name"] == "counter__add"
    assert names == {"counter__add": "counter.add"}
    assert len(register.records) == 2

    register.close_writes()
    with pytest.raises(DomainClosedError):
        register.call("test", "counter.add", {"amount": 1})


def test_serial_key_prevents_parallel_mutation() -> None:
    register = DomainRegister()
    active = 0
    maximum = 0
    guard = threading.Lock()

    def mutate() -> None:
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.01)
        with guard:
            active -= 1

    register.register_function(
        "env",
        "env.step",
        mutate,
        description="Mutate.",
        mutates=True,
        serial_key="env.state",
    )
    threads = [threading.Thread(target=lambda: register.call("test", "env.step")) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert maximum == 1


def test_duplicate_name_is_rejected() -> None:
    register = DomainRegister()
    register.register_reference("one", "shared.value", lambda: 1)
    with pytest.raises(RegisterError):
        register.register_reference("two", "shared.value", lambda: 2)
