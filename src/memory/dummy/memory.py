from __future__ import annotations

from harness import Component, ComponentContext, Function


class DummyMemory(Component):
    name = "memory"

    def __init__(self) -> None:
        self.items: list[dict[str, object]] = []
        self._context: ComponentContext | None = None

    def functions(self):
        return (
            Function(
                "memory.remember",
                "Store one navigation memory item.",
                self._remember,
                input_schema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "position": {"type": "integer"},
                    },
                    "required": ["text", "position"],
                    "additionalProperties": False,
                },
                mutates=True,
                serial_key="memory",
            ),
            Function(
                "memory.search",
                "Return recent navigation memory items.",
                self._search,
                input_schema={
                    "type": "object",
                    "properties": {"limit": {"type": "integer", "minimum": 1}},
                    "required": ["limit"],
                    "additionalProperties": False,
                },
            ),
        )

    async def start(self, context: ComponentContext) -> None:
        self._context = context

    async def _remember(self, _call, arguments):
        item = {"text": arguments["text"], "position": arguments["position"]}
        self.items.append(item)
        return {"stored": len(self.items)}

    async def _search(self, _call, arguments):
        return {"items": self.items[-arguments["limit"] :]}

    async def close(self, reason: str) -> None:
        if self._context is not None:
            self._context.output.set_metadata(items=len(self.items))
            self._context.output.write_json("memory.json", self.items)
            self._context.output.add_artifact("memory.json", "application/json")
