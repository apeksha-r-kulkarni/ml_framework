"""
StatusTracker implementation for runtime telemetry, state tracking, and execution status.

Fulfills Task 7 requirements:
- Thread-safe tracking of execution counts, success/failure counts, and latency statistics.
- Tracking lifecycle state transitions (initialization, execution, pause, resume, reload, unload, shutdown).
- Retention of error details and execution metadata without corruption.
- Public status observability APIs.
"""

import time
import threading
from collections import deque
from typing import Any, Dict, List, Optional, Deque

from ..interfaces.base import BaseStatusTracker
from ..results.execution_result import ExecutionResult, ExecutionContext, ExecutionState


class StatusTracker(BaseStatusTracker):
    """
    Standard thread-safe status and telemetry tracker for the ML Framework.

    Maintains execution metrics, latency percentiles, error logs, and lifecycle
    state history for external observability.
    """

    def __init__(self, max_history: int = 100):
        self.max_history = max(10, int(max_history))
        self._lock = threading.RLock()

        self._start_time = time.time()
        self._total_executions = 0
        self._successful_executions = 0
        self._failed_executions = 0
        self._last_duration_ms = 0.0
        self._total_duration_ms = 0.0
        self._min_duration_ms = float("inf")
        self._max_duration_ms = 0.0
        self._last_execution_time: Optional[float] = None
        self._last_error: Optional[Dict[str, Any]] = None

        self._current_state = ExecutionState.UNINITIALIZED
        self._execution_history: Deque[Dict[str, Any]] = deque(maxlen=self.max_history)
        self._state_history: Deque[Dict[str, Any]] = deque(maxlen=self.max_history)

    def record_success(self, result: ExecutionResult) -> None:
        """
        Records execution success metrics and adds the execution summary to history.
        Thread-safe under lock.
        """
        with self._lock:
            self._total_executions += 1
            self._successful_executions += 1
            duration = float(result.duration_ms)
            self._last_duration_ms = duration
            self._total_duration_ms += duration
            if duration < self._min_duration_ms:
                self._min_duration_ms = duration
            if duration > self._max_duration_ms:
                self._max_duration_ms = duration
            self._last_execution_time = time.time()

            ctx = result.context
            item = {
                "timestamp": self._last_execution_time,
                "correlation_id": ctx.correlation_id if ctx else "unknown",
                "success": True,
                "duration_ms": duration,
                "applied_model_id": result.applied_model_id,
                "stage_durations": dict(ctx.stage_durations) if ctx and ctx.stage_durations else {},
                "error": None,
                "error_type": None,
            }
            self._execution_history.append(item)

    def record_failure(self, error: Exception, context: ExecutionContext) -> None:
        """
        Records execution failure metrics and logs error details.
        Thread-safe under lock.
        """
        with self._lock:
            self._total_executions += 1
            self._failed_executions += 1
            now = time.time()
            self._last_execution_time = now

            err_msg = str(error)
            err_type = type(error).__name__
            corr_id = context.correlation_id if context else "unknown"

            self._last_error = {
                "timestamp": now,
                "correlation_id": corr_id,
                "error_type": err_type,
                "error_message": err_msg,
            }

            item = {
                "timestamp": now,
                "correlation_id": corr_id,
                "success": False,
                "duration_ms": 0.0,
                "applied_model_id": None,
                "stage_durations": dict(context.stage_durations) if context and context.stage_durations else {},
                "error": err_msg,
                "error_type": err_type,
            }
            self._execution_history.append(item)

    def record_state_change(
        self,
        from_state: Any,
        to_state: Any,
        reason: Optional[str] = None
    ) -> None:
        """
        Records a lifecycle state transition.
        Thread-safe under lock.
        """
        with self._lock:
            f_val = from_state.value if hasattr(from_state, "value") else str(from_state)
            t_val = to_state.value if hasattr(to_state, "value") else str(to_state)
            if hasattr(to_state, "value"):
                self._current_state = to_state

            self._state_history.append({
                "timestamp": time.time(),
                "from_state": f_val,
                "to_state": t_val,
                "reason": reason or "",
            })

    def get_status(self) -> Dict[str, Any]:
        """
        Returns a comprehensive snapshot of the tracker's current execution status and metrics.
        Thread-safe under lock.
        """
        with self._lock:
            total = self._total_executions
            avg_duration = (self._total_duration_ms / total) if total > 0 else 0.0
            success_rate = (self._successful_executions / total * 100.0) if total > 0 else 0.0
            min_dur = self._min_duration_ms if self._min_duration_ms != float("inf") else 0.0

            return {
                "current_state": self._current_state.value if hasattr(self._current_state, "value") else str(self._current_state),
                "uptime_seconds": round(time.time() - self._start_time, 2),
                "total_executions": total,
                "successful_executions": self._successful_executions,
                "failed_executions": self._failed_executions,
                "success_rate_percent": round(success_rate, 2),
                "last_duration_ms": round(self._last_duration_ms, 3),
                "avg_duration_ms": round(avg_duration, 3),
                "min_duration_ms": round(min_dur, 3),
                "max_duration_ms": round(self._max_duration_ms, 3),
                "last_execution_time": self._last_execution_time,
                "last_error": dict(self._last_error) if self._last_error else None,
                "recent_executions_count": len(self._execution_history),
                "state_transitions_count": len(self._state_history),
            }

    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Returns recent execution records up to limit."""
        with self._lock:
            hist = list(self._execution_history)
            if limit is not None and limit > 0:
                return hist[-limit:]
            return hist

    def get_state_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Returns recent lifecycle state change records up to limit."""
        with self._lock:
            hist = list(self._state_history)
            if limit is not None and limit > 0:
                return hist[-limit:]
            return hist

    def reset(self) -> None:
        """Resets all metrics and execution history."""
        with self._lock:
            self._total_executions = 0
            self._successful_executions = 0
            self._failed_executions = 0
            self._last_duration_ms = 0.0
            self._total_duration_ms = 0.0
            self._min_duration_ms = float("inf")
            self._max_duration_ms = 0.0
            self._last_execution_time = None
            self._last_error = None
            self._execution_history.clear()
            self._state_history.clear()
