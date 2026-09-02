class HarnessError(RuntimeError):
    """Base error for configuration and runtime contract failures."""


class RegisterError(HarnessError):
    """Raised when a Domain register operation violates its contract."""


class DomainClosedError(RegisterError):
    """Raised when a mutation is attempted after Domain termination."""
