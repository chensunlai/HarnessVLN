class NexusError(Exception):
    """Base error for the Nexus runtime."""


class ContractError(NexusError):
    """A component or function violates a static runtime contract."""


class FunctionError(NexusError):
    """Base error for function registration and invocation."""


class FunctionNotFoundError(FunctionError):
    pass


class FunctionPermissionError(FunctionError):
    pass


class FunctionValidationError(FunctionError):
    pass


class DomainClosedError(FunctionError):
    pass


class DomainExecutionError(NexusError):
    pass
