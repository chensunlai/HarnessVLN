from __future__ import annotations

import importlib
from typing import TypeVar

from harness.config import FactorySpec
from harness.errors import ContractError


T = TypeVar("T")


def instantiate(spec: FactorySpec, expected: type[T], role: str) -> T:
    module_name, attribute_name = spec.target.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        raise ContractError(
            f"cannot import module {module_name!r} for {role}: {type(error).__name__}: {error}"
        ) from error
    try:
        factory = getattr(module, attribute_name)
    except AttributeError as error:
        raise ContractError(
            f"module {module_name!r} has no factory {attribute_name!r} for {role}"
        ) from error
    if not callable(factory):
        raise ContractError(f"factory {spec.target!r} for {role} is not callable")
    try:
        value = factory(**spec.parameters)
    except Exception as error:
        raise ContractError(
            f"factory {spec.target!r} for {role} failed: {type(error).__name__}: {error}"
        ) from error
    if not isinstance(value, expected):
        raise ContractError(
            f"factory {spec.target!r} for {role} returned {type(value).__name__}, "
            f"expected {expected.__name__}"
        )
    return value
