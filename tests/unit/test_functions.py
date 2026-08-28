import asyncio

import pytest

from harness.errors import (
    ContractError,
    DomainClosedError,
    FunctionPermissionError,
    FunctionValidationError,
)
from harness.functions import Function, FunctionBus


VALUE_INPUT = {
    "type": "object",
    "properties": {"value": {"type": "integer"}},
    "required": ["value"],
    "additionalProperties": False,
}
VALUE_OUTPUT = {
    "type": "object",
    "properties": {"value": {"type": "integer"}},
    "required": ["value"],
    "additionalProperties": False,
}


def run(operation):
    return asyncio.run(operation)


def test_function_bus_exposes_only_allowed_model_tools_and_validates_schema():
    async def scenario():
        async def echo(_call, arguments):
            return {"value": arguments["value"]}

        bus = FunctionBus()
        bus.register(
            "provider",
            (
                Function(
                    name="demo.echo",
                    description="Echo an integer.",
                    handler=echo,
                    input_schema=VALUE_INPUT,
                    output_schema=VALUE_OUTPUT,
                ),
                Function(
                    name="private.read",
                    description="Return private state.",
                    handler=echo,
                    input_schema=VALUE_INPUT,
                    output_schema=VALUE_OUTPUT,
                ),
            ),
        )
        bus.seal()
        client = bus.client("agent", {"demo.echo"})

        assert [spec.name for spec in client.specs] == ["demo.echo"]
        assert client.specs[0].as_model_tool()["parameters"] == VALUE_INPUT
        assert await client.call("demo.echo", value=3) == {"value": 3}
        with pytest.raises(FunctionPermissionError):
            await client.call("private.read", value=3)
        with pytest.raises(FunctionValidationError):
            await client.call("demo.echo", value="three")
        assert [record.outcome for record in bus.records] == ["ok"]

    run(scenario())


def test_duplicate_functions_and_missing_requirements_fail_before_execution():
    async def handler(_call, _arguments):
        return {}

    bus = FunctionBus()
    function = Function("demo.call", "Do work.", handler)
    bus.register("first", (function,))
    with pytest.raises(ContractError, match="provided by both"):
        bus.register("second", (function,))
    with pytest.raises(ContractError, match="unavailable"):
        bus.require("agent", {"missing.call"})


def test_mutations_with_the_same_resource_key_are_serialized():
    async def scenario():
        active = 0
        peak = 0

        async def mutate(_call, arguments):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return {"value": arguments["value"]}

        bus = FunctionBus()
        bus.register(
            "environment",
            (
                Function(
                    "nav.move",
                    "Move in the environment.",
                    mutate,
                    input_schema=VALUE_INPUT,
                    output_schema=VALUE_OUTPUT,
                    mutates=True,
                    serial_key="environment",
                ),
            ),
        )
        bus.seal()
        client = bus.client("agent", {"nav.move"})
        values = await asyncio.gather(
            client.call("nav.move", value=1),
            client.call("nav.move", value=2),
        )
        assert values == [{"value": 1}, {"value": 2}]
        assert peak == 1

    run(scenario())


def test_bus_does_not_report_idle_during_a_nested_function_call():
    async def scenario():
        inner_finished = asyncio.Event()
        release_outer = asyncio.Event()
        bus = FunctionBus()
        system = bus.client("placeholder", None)

        async def inner(_call, _arguments):
            return {"value": 1}

        async def outer(_call, _arguments):
            await system.call("demo.inner")
            inner_finished.set()
            await release_outer.wait()
            return {"value": 2}

        bus.register(
            "provider",
            (
                Function("demo.inner", "Inner call.", inner),
                Function("demo.outer", "Outer call.", outer),
            ),
        )
        bus.seal()
        system = bus.client("system", None)
        invocation = asyncio.create_task(system.call("demo.outer"))
        await inner_finished.wait()
        closing = asyncio.create_task(bus.close())
        await asyncio.sleep(0)
        assert not closing.done()
        release_outer.set()
        assert await invocation == {"value": 2}
        await closing

    run(scenario())


def test_write_barrier_rejects_new_mutations_but_keeps_reads_available():
    async def scenario():
        async def handler(_call, _arguments):
            return {}

        bus = FunctionBus()
        bus.register(
            "environment",
            (
                Function("nav.observe", "Observe.", handler),
                Function(
                    "nav.move",
                    "Move.",
                    handler,
                    mutates=True,
                    serial_key="environment",
                ),
            ),
        )
        bus.seal()
        client = bus.client("agent", {"nav.observe", "nav.move"})
        bus.close_writes()
        assert await client.call("nav.observe") == {}
        with pytest.raises(DomainClosedError):
            await client.call("nav.move")

    run(scenario())
