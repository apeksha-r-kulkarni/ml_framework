"""
Unit tests for the manual model test runner (tests/manual/model_test_runner.py).

Verifies CLI argument validation, lifecycle orchestration, mock model loading,
inference failure handling, cleanup guarantees, output formatting, and error reporting.
"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml_framework import ExecutionState, ExecutionResult
from ml_framework.errors import ModelLoadError, InferenceError, ModelFormatError
from tests.manual.model_test_runner import (
    run_model_test,
    build_cli_parser,
    format_model_output,
    main,
    ModelTestReport,
)


class TestManualModelTestRunner(unittest.TestCase):
    """Unit test suite for model_test_runner CLI and orchestration logic."""

    def test_cli_argument_parsing(self):
        """Test argument parsing across positional, flag, input, and configuration options."""
        parser = build_cli_parser()

        # 1. Positional model path with -i local file
        args1 = parser.parse_args(["/path/to/model", "-i", "/path/to/file.txt", "-f", "spacy"])
        self.assertEqual(args1.model_path_pos, "/path/to/model")
        self.assertIsNone(args1.model_path_opt)
        self.assertEqual(args1.input_source, "/path/to/file.txt")
        self.assertEqual(args1.framework, "spacy")
        self.assertEqual(args1.device, "auto")

        # 2. Flag model path with --input-file storage URI
        args2 = parser.parse_args(["-m", "/flag/model", "--input-file", "s3://bucket/test.txt", "-d", "cpu", "--timeout", "15.0", "-v", "--json"])
        self.assertEqual(args2.model_path_opt, "/flag/model")
        self.assertIsNone(args2.model_path_pos)
        self.assertEqual(args2.input_source, "s3://bucket/test.txt")
        self.assertEqual(args2.device, "cpu")
        self.assertEqual(args2.timeout, 15.0)
        self.assertTrue(args2.verbose)
        self.assertTrue(args2.json)

        # 3. Direct string flags (--text / -t) must be rejected as unrecognized
        with self.assertRaises(SystemExit):
            parser.parse_args(["/model", "-t", "sample text"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["/model", "--text", "sample text"])

    def test_main_missing_model_path(self):
        """Test main returns exit code 2 when model path is omitted."""
        code = main([])
        self.assertEqual(code, 2)

    def test_main_missing_input_source(self):
        """Test main returns exit code 2 when input source is omitted."""
        code = main(["/path/to/model"])
        self.assertEqual(code, 2)

    def test_run_model_test_missing_model_path_validation(self):
        """Test run_model_test validates empty model_path string."""
        report = run_model_test(model_path="   ", input_source="/dummy/file.txt", print_output=False)
        self.assertFalse(report.success)
        self.assertEqual(report.exit_code, 2)
        self.assertIn("Model path cannot be empty", report.error)

    def test_run_model_test_missing_input_source(self):
        """Test run_model_test returns exit code 2 when no input source is specified."""
        report = run_model_test(model_path="/dummy/path", print_output=False)
        self.assertFalse(report.success)
        self.assertEqual(report.exit_code, 2)
        self.assertIn("Input source is required", report.error)

    def test_run_model_test_missing_input_file(self):
        """Test run_model_test handles non-existent input file path."""
        report = run_model_test(
            model_path="/dummy/path",
            input_file="/non/existent/test_file.txt",
            print_output=False,
        )
        self.assertFalse(report.success)
        self.assertEqual(report.exit_code, 1)
        self.assertIn("does not exist", report.error)

    def test_run_model_test_direct_string_rejection(self):
        """Test run_model_test explicitly rejects unsupported standalone direct string inputs."""
        report = run_model_test(
            model_path="/dummy/path",
            input_source="Apple hired John Smith in London.",
            print_output=False,
        )
        self.assertFalse(report.success)
        self.assertEqual(report.exit_code, 1)
        self.assertEqual(report.error_type, "InvalidInputSourceError")
        self.assertIn("Direct string input is not supported", report.error)

        # Legacy keyword argument input_text must also be explicitly rejected
        report_legacy = run_model_test(
            model_path="/dummy/path",
            input_text="Sample direct string text",
            print_output=False,
        )
        self.assertFalse(report_legacy.success)
        self.assertEqual(report_legacy.exit_code, 1)
        self.assertEqual(report_legacy.error_type, "InvalidInputSourceError")
        self.assertIn("Direct string input is not supported", report_legacy.error)

    def test_run_model_test_successful_mocked(self):
        """Test end-to-end model test runner lifecycle with mocked framework execution using a local file."""
        mock_manager = MagicMock()
        mock_manager.state = ExecutionState.READY
        mock_manager.is_ready = True
        mock_adapter = MagicMock()
        mock_adapter.architecture = "MockNERArchitecture"
        type(mock_adapter).__name__ = "MockModelAdapter"
        mock_manager._active_model_instance = mock_adapter

        mock_manager.process.return_value = ExecutionResult(
            success=True,
            output={"entities": [{"entity": "ORG", "word": "Apple", "score": 0.98, "start": 0, "end": 5}]},
            duration_ms=12.5,
        )

        with tempfile.NamedTemporaryFile("w", delete=False) as tf:
            tf.write("Apple hired John.")
            temp_path = tf.name

        try:
            with patch("ml_framework.ml.load", return_value=mock_manager) as mock_ml_load:
                report = run_model_test(
                    model_path="/mock/model/path",
                    input_source=temp_path,
                    framework="transformers",
                    device="cpu",
                    print_output=False,
                )

                self.assertTrue(report.success)
                self.assertEqual(report.exit_code, 0)
                self.assertEqual(report.model_path, "/mock/model/path")
                self.assertEqual(report.framework, "transformers")
                self.assertEqual(report.adapter_class, "MockModelAdapter")
                self.assertEqual(report.detected_architecture, "MockNERArchitecture")
                self.assertGreater(report.load_duration_ms, 0.0)
                self.assertGreater(report.inference_duration_ms, 0.0)
                self.assertEqual(len(report.entities), 1)
                self.assertEqual(report.entities[0]["word"], "Apple")

                # Verify lifecycle calls
                mock_ml_load.assert_called_once_with(
                    model_source="/mock/model/path",
                    framework="transformers",
                    device="cpu",
                    timeout_seconds=30.0,
                )
                mock_manager.process.assert_called_once_with(temp_path)
                mock_manager.unload.assert_called_once()
                mock_manager.stop.assert_called_once()
                self.assertTrue(report.cleanup_successful)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_run_model_test_minio_and_kafka_sources_mocked(self):
        """Test manual runner accepts MinIO and Kafka URI input sources."""
        mock_manager = MagicMock()
        mock_manager.state = ExecutionState.READY
        mock_manager.is_ready = True
        mock_manager.process.return_value = ExecutionResult(success=True, output={"status": "ok"})

        with patch("ml_framework.ml.load", return_value=mock_manager):
            # 1. MinIO / S3 source
            report_minio = run_model_test(
                model_path="/mock/model",
                input_source="s3://test-bucket/test-file.txt",
                print_output=False,
            )
            self.assertTrue(report_minio.success)
            self.assertEqual(report_minio.input_source_type, "minio")
            mock_manager.process.assert_called_with("s3://test-bucket/test-file.txt")

            # 2. Kafka source
            report_kafka = run_model_test(
                model_path="/mock/model",
                input_source="kafka://test-cluster-topic",
                print_output=False,
            )
            self.assertTrue(report_kafka.success)
            self.assertEqual(report_kafka.input_source_type, "kafka")
            mock_manager.process.assert_called_with("kafka://test-cluster-topic")

    def test_run_model_test_load_failure(self):
        """Test model load failure reports error and returns exit code 2."""
        with patch("ml_framework.ml.load", side_effect=ModelLoadError("Simulated artifact missing error")):
            report = run_model_test(
                model_path="/broken/model",
                input_source="s3://bucket/test.txt",
                print_output=False,
            )
            self.assertFalse(report.success)
            self.assertEqual(report.exit_code, 2)
            self.assertEqual(report.error_type, "ModelLoadError")
            self.assertIn("Simulated artifact missing error", report.error)

    def test_run_model_test_inference_failure_cleanup(self):
        """Test that manager.unload and stop are strictly called when inference reports failure."""
        mock_manager = MagicMock()
        mock_manager.state = ExecutionState.READY
        mock_manager.is_ready = True

        mock_manager.process.return_value = ExecutionResult(
            success=False,
            output=None,
            error="Model memory exhausted",
            error_type="InferenceError",
        )

        with patch("ml_framework.ml.load", return_value=mock_manager):
            report = run_model_test(
                model_path="/mock/model",
                input_source="s3://bucket/test.txt",
                print_output=False,
            )
            self.assertFalse(report.success)
            self.assertEqual(report.exit_code, 1)
            self.assertEqual(report.error, "Model memory exhausted")
            # Verify cleanup occurred in finally block
            mock_manager.unload.assert_called_once()
            mock_manager.stop.assert_called_once()
            self.assertTrue(report.cleanup_successful)

    def test_run_model_test_inference_exception_cleanup(self):
        """Test that manager.unload and stop are strictly called when inference raises an unexpected exception."""
        mock_manager = MagicMock()
        mock_manager.state = ExecutionState.READY
        mock_manager.is_ready = True
        mock_manager.process.side_effect = RuntimeError("Critical compute device crash")

        with patch("ml_framework.ml.load", return_value=mock_manager):
            report = run_model_test(
                model_path="/mock/model",
                input_source="s3://bucket/test.txt",
                print_output=False,
            )
            self.assertFalse(report.success)
            self.assertEqual(report.exit_code, 1)
            self.assertEqual(report.error_type, "RuntimeError")
            self.assertIn("Critical compute device crash", report.error)
            mock_manager.unload.assert_called_once()
            mock_manager.stop.assert_called_once()

    def test_run_model_test_with_local_input_file(self):
        """Test reading input from an actual local input file."""
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("Text inside local file for inference.")
            temp_path = f.name

        try:
            mock_manager = MagicMock()
            mock_manager.state = ExecutionState.READY
            mock_manager.is_ready = True
            mock_manager.process.return_value = ExecutionResult(success=True, output="Processed text file")

            with patch("ml_framework.ml.load", return_value=mock_manager):
                report = run_model_test(
                    model_path="/mock/model",
                    input_file=temp_path,
                    print_output=False,
                )
                self.assertTrue(report.success)
                self.assertEqual(report.input_source_type, "file")
                mock_manager.process.assert_called_once_with(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_format_model_output_spacy(self):
        """Test output formatting for spaCy Doc representation."""
        mock_doc = MagicMock()
        mock_doc.text = "Apple hired John Smith in London."

        mock_ent1 = MagicMock(text="Apple", label_="ORG", start_char=0, end_char=5)
        mock_ent2 = MagicMock(text="John Smith", label_="PERSON", start_char=12, end_char=22)
        mock_doc.ents = [mock_ent1, mock_ent2]

        mock_tok1 = MagicMock(text="Apple")
        mock_tok2 = MagicMock(text="hired")
        mock_doc.__iter__.return_value = [mock_tok1, mock_tok2]

        formatted = format_model_output(mock_doc)
        self.assertIn("spaCy Doc", formatted)
        self.assertIn("Apple hired John Smith in London.", formatted)
        self.assertIn("ORG", formatted)
        self.assertIn("PERSON", formatted)
        self.assertIn("Apple", formatted)
        self.assertIn("John Smith", formatted)

    def test_format_model_output_transformers(self):
        """Test output formatting for Transformers token classification dictionary."""
        tf_out = {
            "text": "Apple hired John Smith.",
            "tokens": [
                {"token": "Apple", "label": "B-ORG", "class_id": 3, "score": 0.98, "start": 0, "end": 5},
                {"token": "John", "label": "B-PER", "class_id": 1, "score": 0.99, "start": 12, "end": 16},
            ],
            "entities": [
                {"entity": "ORG", "word": "Apple", "score": 0.98, "start": 0, "end": 5},
                {"entity": "PER", "word": "John", "score": 0.99, "start": 12, "end": 16},
            ],
        }
        formatted = format_model_output(tf_out, verbose=True)
        self.assertIn("Transformers Token Classification", formatted)
        self.assertIn("ORG", formatted)
        self.assertIn("PER", formatted)
        self.assertIn("0.9800", formatted)
        self.assertIn("B-ORG", formatted)

    def test_format_model_output_sequence_classification(self):
        """Test output formatting for sequence classification dictionary."""
        seq_out = {
            "label": "POSITIVE",
            "score": 0.9754,
            "class_id": 1,
            "probabilities": {"NEGATIVE": 0.0246, "POSITIVE": 0.9754},
        }
        formatted = format_model_output(seq_out)
        self.assertIn("Sequence Classification", formatted)
        self.assertIn("POSITIVE", formatted)
        self.assertIn("0.9754", formatted)
        self.assertIn("NEGATIVE", formatted)

    def test_format_model_output_question_answering(self):
        """Test output formatting for question answering dictionary."""
        qa_out = {
            "answer": "London",
            "score": 0.9521,
            "start": 26,
            "end": 32,
        }
        formatted = format_model_output(qa_out)
        self.assertIn("Question Answering", formatted)
        self.assertIn("London", formatted)
        self.assertIn("0.9521", formatted)
        self.assertIn("26", formatted)

    def test_json_cli_mode(self):
        """Test that --json flag outputs valid JSON dictionary with exit code 0."""
        with tempfile.NamedTemporaryFile("w", delete=False) as tf:
            tf.write("Sample test payload")
            temp_path = tf.name

        try:
            mock_manager = MagicMock()
            mock_manager.state = ExecutionState.READY
            mock_manager.is_ready = True
            mock_manager.process.return_value = ExecutionResult(success=True, output={"status": "ok"})

            with patch("ml_framework.ml.load", return_value=mock_manager):
                import io
                captured_stdout = io.StringIO()
                with patch("sys.stdout", captured_stdout):
                    code = main(["/mock/model", "-i", temp_path, "--json"])

                self.assertEqual(code, 0)
                json_str = captured_stdout.getvalue()
                parsed = json.loads(json_str)
                self.assertTrue(parsed["success"])
                self.assertEqual(parsed["model_path"], "/mock/model")
                self.assertEqual(parsed["output"], {"status": "ok"})
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
