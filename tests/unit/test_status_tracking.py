"""
Unit tests for StatusTracker and ExecutionManager status observability (Task 7).
"""

import sys
import unittest
import threading
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ml_framework.status.tracker import StatusTracker
from ml_framework.execution_manager import ExecutionManager
from ml_framework.config.model_config import FrameworkConfig
from ml_framework.results.execution_result import ExecutionResult, ExecutionContext, ExecutionState


class TestStatusTracker(unittest.TestCase):
    """Test suite for StatusTracker metrics, state history, concurrency, and observability."""

    def setUp(self):
        self.tracker = StatusTracker(max_history=50)

    def test_record_success_updates_metrics_and_history(self):
        """Test recording success updates metrics correctly."""
        ctx = ExecutionContext(correlation_id="corr-success-1")
        ctx.record_stage("inference", 12.5)
        res = ExecutionResult(
            success=True,
            output="test output",
            duration_ms=15.0,
            context=ctx,
            state=ExecutionState.READY,
            applied_model_id="test_model:1.0",
        )
        self.tracker.record_success(res)

        status = self.tracker.get_status()
        self.assertEqual(status["total_executions"], 1)
        self.assertEqual(status["successful_executions"], 1)
        self.assertEqual(status["failed_executions"], 0)
        self.assertEqual(status["success_rate_percent"], 100.0)
        self.assertEqual(status["last_duration_ms"], 15.0)
        self.assertEqual(status["avg_duration_ms"], 15.0)

        history = self.tracker.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["correlation_id"], "corr-success-1")
        self.assertTrue(history[0]["success"])
        self.assertEqual(history[0]["duration_ms"], 15.0)

    def test_record_failure_updates_metrics_and_last_error(self):
        """Test recording failure updates error tracking."""
        ctx = ExecutionContext(correlation_id="corr-fail-1")
        err = ValueError("Invalid input tensor")
        self.tracker.record_failure(err, ctx)

        status = self.tracker.get_status()
        self.assertEqual(status["total_executions"], 1)
        self.assertEqual(status["successful_executions"], 0)
        self.assertEqual(status["failed_executions"], 1)
        self.assertEqual(status["success_rate_percent"], 0.0)
        self.assertIsNotNone(status["last_error"])
        self.assertEqual(status["last_error"]["error_type"], "ValueError")
        self.assertEqual(status["last_error"]["correlation_id"], "corr-fail-1")

    def test_state_change_tracking(self):
        """Test recording lifecycle state changes."""
        self.tracker.record_state_change(ExecutionState.UNINITIALIZED, ExecutionState.INITIALIZING, "start init")
        self.tracker.record_state_change(ExecutionState.INITIALIZING, ExecutionState.READY, "init complete")

        state_hist = self.tracker.get_state_history()
        self.assertEqual(len(state_hist), 2)
        self.assertEqual(state_hist[0]["from_state"], "UNINITIALIZED")
        self.assertEqual(state_hist[0]["to_state"], "INITIALIZING")
        self.assertEqual(state_hist[1]["from_state"], "INITIALIZING")
        self.assertEqual(state_hist[1]["to_state"], "READY")

    def test_reset_clears_metrics(self):
        """Test resetting tracker restores zeroed metrics and clears history."""
        ctx = ExecutionContext(correlation_id="test-reset")
        res = ExecutionResult(success=True, duration_ms=10.0, context=ctx)
        self.tracker.record_success(res)
        self.assertEqual(self.tracker.get_status()["total_executions"], 1)

        self.tracker.reset()
        status = self.tracker.get_status()
        self.assertEqual(status["total_executions"], 0)
        self.assertEqual(len(self.tracker.get_history()), 0)

    def test_concurrent_recording_thread_safety(self):
        """Test recording successes and failures concurrently across multiple threads."""
        threads = []
        num_threads = 8
        records_per_thread = 50

        def worker(thread_idx: int):
            for i in range(records_per_thread):
                if (thread_idx + i) % 2 == 0:
                    ctx = ExecutionContext(correlation_id=f"t{thread_idx}-{i}")
                    res = ExecutionResult(success=True, duration_ms=5.0, context=ctx)
                    self.tracker.record_success(res)
                else:
                    ctx = ExecutionContext(correlation_id=f"t{thread_idx}-{i}")
                    self.tracker.record_failure(RuntimeError(f"error-{i}"), ctx)

        for t_idx in range(num_threads):
            t = threading.Thread(target=worker, args=(t_idx,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        status = self.tracker.get_status()
        total_expected = num_threads * records_per_thread
        self.assertEqual(status["total_executions"], total_expected)
        self.assertEqual(
            status["successful_executions"] + status["failed_executions"],
            total_expected
        )

    def test_execution_manager_status_observability(self):
        """Test that ExecutionManager exposes observable status and records lifecycle events."""
        manager = ExecutionManager()
        self.assertEqual(manager.state, ExecutionState.UNINITIALIZED)
        self.assertEqual(manager.status["current_state"], "UNINITIALIZED")

        # Initialize
        manager.initialize()
        self.assertEqual(manager.state, ExecutionState.READY)
        self.assertEqual(manager.status["current_state"], "READY")

        # Execute
        result = manager.execute("sample telemetry test")
        self.assertTrue(result.success)
        self.assertEqual(manager.status["total_executions"], 1)
        self.assertEqual(manager.status["successful_executions"], 1)

        # Pause and Resume
        manager.pause()
        self.assertEqual(manager.state, ExecutionState.PAUSED)
        self.assertEqual(manager.status["current_state"], "PAUSED")

        manager.resume()
        self.assertEqual(manager.state, ExecutionState.READY)
        self.assertEqual(manager.status["current_state"], "READY")

        # Shutdown
        manager.shutdown()
        self.assertEqual(manager.state, ExecutionState.STOPPED)
        self.assertEqual(manager.status["current_state"], "STOPPED")

        # Check state transition history
        state_hist = manager.status_tracker.get_state_history()
        self.assertGreaterEqual(len(state_hist), 4)


if __name__ == "__main__":
    unittest.main()
