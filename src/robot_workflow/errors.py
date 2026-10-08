"""Domain-specific failures with safe, explicit semantics."""


class WorkflowError(RuntimeError):
    """Base class for workflow failures."""


class ValidationError(WorkflowError):
    """Input or domain data failed validation."""


class SnapshotValidationError(ValidationError):
    """A camera snapshot is incomplete, stale, or uncalibrated."""


class TransitionError(WorkflowError):
    """A state transition is not permitted."""


class SafetyViolation(WorkflowError):
    """A safety invariant was not satisfied."""


class ApprovalError(SafetyViolation):
    """Approval is missing, stale, already consumed, or does not match a plan."""


class ServiceError(WorkflowError):
    """A model or planning service failed."""


class ProtocolError(ServiceError):
    """A remote service violated the length-prefixed JSON protocol."""
