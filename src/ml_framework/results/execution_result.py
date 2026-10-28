"""
Result and context definitions for the ML Framework.

Provides typed encapsulation for execution context, lifecycle state tracking,
and structured outputs as defined in the Successful Execution and Execution Flow diagrams.
"""

from dataclasses import dataclass, field
from enum import Enum
import time
import uuid
from typing import Dict, Any, Optional


class ExecutionState(Enum):
    """Lifecycle states of the ExecutionManager."""
    UNINITIALIZED = "UNINITIALIZED"
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    EXECUTING = "EXECUTING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    FAILED = "FAILED"
    DEGRADED = "DEGRADED"
    RELOADING = "RELOADING"



@dataclass
class ExecutionContext:
    """
    Contextual metadata carried across all execution stages.
    Enables end-to-end tracing, correlation ID assignment, and per-stage latency tracking.
    """
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    stage_durations: Dict[str, float] = field(default_factory=dict)

    def record_stage(self, stage_name: str, duration_ms: float) -> None:
        """Records execution duration in milliseconds for an individual stage."""
        self.stage_durations[stage_name] = round(duration_ms, 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "stage_durations": self.stage_durations,
        }


@dataclass
class ExecutionResult:
    """
    Standardized result object returned by ExecutionManager.execute().
    Guarantees a consistent return contract regardless of underlying model or modality.
    """
    success: bool
    output: Any = None
    duration_ms: float = 0.0
    context: ExecutionContext = field(default_factory=ExecutionContext)
    state: ExecutionState = ExecutionState.READY
    error: Optional[str] = None
    error_type: Optional[str] = None
    applied_model_id: Optional[str] = None

    def is_success(self) -> bool:
        return self.success

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "duration_ms": self.duration_ms,
            "applied_model_id": self.applied_model_id,
            "state": self.state.value if isinstance(self.state, ExecutionState) else str(self.state),
            "error": self.error,
            "error_type": self.error_type,
            "context": self.context.to_dict() if self.context else None,
        }
