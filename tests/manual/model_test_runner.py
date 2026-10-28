"""
Reusable Manual Model Testing Runner for ml_framework.

Provides CLI and programmatic interfaces to load, execute, and verify
different local model artifacts (spaCy, Transformers, ONNX, Speech)
exclusively through the official public ml_framework API.
"""

import os
import sys
import time
import json
import logging
import argparse
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure ml_framework src directory is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml_framework import ml, ExecutionState
from ml_framework.config.model_config import StorageConfig
from ml_framework.models.format_detector import FormatDetector
from ml_framework.errors import (
    MLFrameworkError,
    ModelLoadError,
    ModelFormatError,
    EnvironmentError,
    DependencyMissingError,
    HardwareIncompatibleError,
    InputProcessingError,
    InvalidInputSourceError,
    InferenceError,
    StateError,
    ConfigurationError,
)

logger = logging.getLogger("ml_framework.manual_runner")


def _load_env_defaults() -> None:
    """Explicitly loads local .env variables into os.environ if present."""
    env_file = Path(__file__).resolve().parent.parent.parent / ".env"
    if not env_file.is_file():
        return
    try:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("export "):
                    line = line[7:].strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip()
                if len(v) >= 2 and ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
                    v = v[1:-1]
                elif " #" in v:
                    v = v.split(" #", 1)[0].rstrip()
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


_load_env_defaults()


@dataclass
class ModelTestReport:
    """Structured report returned by the model test runner."""
    success: bool
    model_path: str
    status: str = "FAIL"  # PASS, PARTIAL, BLOCKED, FAIL
    stage_reached: str = "init"
    blocker_reason: Optional[str] = None
    framework: str = "auto"
    detected_framework: Optional[str] = None
    detected_architecture: Optional[str] = None
    adapter_class: Optional[str] = None
    model_state: Optional[str] = None
    input_source_type: Optional[str] = None
    input_data_summary: Optional[str] = None
    load_duration_ms: float = 0.0
    inference_duration_ms: float = 0.0
    total_duration_ms: float = 0.0
    raw_output: Any = None
    formatted_output: Optional[str] = None
    entities: List[Dict[str, Any]] = field(default_factory=list)
    tokens: List[Dict[str, Any]] = field(default_factory=list)
    cleanup_successful: bool = False
    error: Optional[str] = None
    error_type: Optional[str] = None
    exit_code: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Converts report into a clean, JSON-serializable dictionary."""
        serializable_output = self._serialize_output(self.raw_output)
        return {
            "success": self.success,
            "status": self.status,
            "stage_reached": self.stage_reached,
            "blocker_reason": self.blocker_reason,
            "exit_code": self.exit_code,
            "model_path": self.model_path,
            "framework": self.framework,
            "detected_framework": self.detected_framework,
            "detected_architecture": self.detected_architecture,
            "adapter_class": self.adapter_class,
            "model_state": self.model_state,
            "input_source_type": self.input_source_type,
            "input_data_summary": self.input_data_summary,
            "timing_ms": {
                "load": round(self.load_duration_ms, 2),
                "inference": round(self.inference_duration_ms, 2),
                "total": round(self.total_duration_ms, 2),
            },
            "output_summary": {
                "entities_count": len(self.entities),
                "tokens_count": len(self.tokens),
            },
            "entities": self.entities,
            "output": serializable_output,
            "cleanup_successful": self.cleanup_successful,
            "error": self.error,
            "error_type": self.error_type,
        }

    @staticmethod
    def _serialize_output(output: Any) -> Any:
        """Safely transforms complex objects (like spacy.tokens.Doc) into plain serializable structures."""
        if output is None:
            return None
        # spaCy Doc
        if hasattr(output, "ents") and hasattr(output, "text"):
            return {
                "type": "spacy_doc",
                "text": output.text,
                "entities": [
                    {
                        "entity": ent.label_,
                        "word": ent.text,
                        "start": ent.start_char,
                        "end": ent.end_char,
                    }
                    for ent in output.ents
                ],
                "tokens": [t.text for t in output],
            }
        if isinstance(output, (dict, list, str, int, float, bool)):
            return output
        return str(output)


def download_input_artifact(
    bucket: str,
    key: str,
    storage_config: Optional[Any] = None,
) -> Tuple[str, str]:
    """
    Downloads an input artifact from MinIO / S3 into a temporary staging file,
    preserving the original file extension.

    Returns:
        Tuple of (staged_file_path, temp_directory_to_cleanup)
    """
    import boto3
    from botocore.config import Config

    storage_cfg = storage_config or StorageConfig.from_env()
    boto_cfg = Config(
        signature_version="s3v4",
        s3={"addressing_style": "path"},
        retries={"max_attempts": 10, "mode": "adaptive"},
        connect_timeout=30,
        read_timeout=120,
    )
    s3 = boto3.client(
        "s3",
        endpoint_url=storage_cfg.endpoint_url,
        aws_access_key_id=storage_cfg.access_key,
        aws_secret_access_key=storage_cfg.secret_key,
        region_name=storage_cfg.region,
        use_ssl=storage_cfg.secure,
        verify=storage_cfg.verify_ssl,
        config=boto_cfg,
    )

    clean_key = key.lstrip("/")
    ext = os.path.splitext(clean_key)[1]
    base_name = os.path.basename(clean_key).replace(" ", "_")
    if not ext:
        ext = ".bin"

    temp_dir = tempfile.mkdtemp(prefix="ml_framework_input_")
    dest_path = os.path.join(temp_dir, f"input_{base_name}")
    s3.download_file(Bucket=bucket, Key=key, Filename=dest_path)
    return dest_path, temp_dir


def format_model_output(output: Any, verbose: bool = False) -> str:
    """
    Renders raw model execution output into a clean, readable text presentation.
    Supports spaCy Doc, Transformers Token Classification, Sequence Classification,
    Question Answering, and generic payloads without assuming uniform structure.
    """
    lines = []

    # 1. spaCy Doc Output
    if hasattr(output, "ents") and hasattr(output, "text"):
        lines.append("-" * 70)
        lines.append("OUTPUT TYPE: spaCy Doc")
        lines.append("-" * 70)
        lines.append(f"Text: \"{output.text}\"")
        lines.append("")

        ents = list(output.ents)
        if ents:
            lines.append(f"Detected Entities ({len(ents)} found):")
            for idx, ent in enumerate(ents, start=1):
                lines.append(
                    f"  #{idx:<2} {ent.label_:<12} \"{ent.text}\" (Span: {ent.start_char}-{ent.end_char})"
                )
        else:
            lines.append("Detected Entities: None")

        tokens = [t.text for t in output]
        if verbose or len(tokens) <= 20:
            lines.append(f"\nTokens ({len(tokens)} total): {tokens}")
        else:
            lines.append(f"\nTokens ({len(tokens)} total, first 15 shown): {tokens[:15]}...")

        return "\n".join(lines)

    # 2. Transformers Token Classification Output Dict
    if isinstance(output, dict) and "tokens" in output and "entities" in output:
        lines.append("-" * 70)
        lines.append("OUTPUT TYPE: Transformers Token Classification")
        lines.append("-" * 70)
        text = output.get("text", "")
        if text:
            lines.append(f"Text: \"{text}\"")
            lines.append("")

        entities = output.get("entities", [])
        if entities:
            lines.append(f"Grouped Entities ({len(entities)} found):")
            for idx, ent in enumerate(entities, start=1):
                ent_type = ent.get("entity", "")
                word = ent.get("word", "")
                score = ent.get("score")
                score_str = f"Score: {score:.4f}" if isinstance(score, (int, float)) else ""
                span_str = ""
                if "start" in ent and "end" in ent and ent["start"] is not None and ent["end"] is not None:
                    span_str = f"Span: {ent['start']}-{ent['end']}"
                details = "  ".join(filter(None, [score_str, span_str]))
                lines.append(f"  #{idx:<2} {ent_type:<10} \"{word}\" ({details})")
        else:
            lines.append("Grouped Entities: None")

        raw_tokens = output.get("tokens", [])
        if raw_tokens:
            if verbose or len(raw_tokens) <= 12:
                lines.append(f"\nToken-Level Predictions ({len(raw_tokens)} total):")
                for idx, t in enumerate(raw_tokens, start=1):
                    tok_name = t.get("token", "")
                    tok_lbl = t.get("label", "")
                    cid = t.get("class_id", "")
                    score = t.get("score", "")
                    span = f"({t.get('start')}, {t.get('end')})"
                    lines.append(
                        f"  #{idx:<2} {tok_name:<12} Label: {tok_lbl:<10} ClassID: {cid:<3} Score: {score:<7} Span: {span}"
                    )
            else:
                lines.append(f"\nToken-Level Predictions: {len(raw_tokens)} tokens (run with --verbose to view all)")

        return "\n".join(lines)

    # 2.5. Speech Output Dict (Speech-to-Text or Language Identification)
    if isinstance(output, dict) and output.get("task") in ("speech_to_text", "speech_seq2seq"):
        lines.append("-" * 70)
        lines.append("OUTPUT TYPE: Speech Recognition (Speech-to-Text)")
        lines.append("-" * 70)
        lines.append(f"Transcription   : \"{output.get('text', '')}\"")
        lines.append(f"Audio Duration  : {output.get('duration_seconds', 0.0):.2f} s")
        lines.append(f"Sampling Rate   : {output.get('sampling_rate', 16000)} Hz")
        lines.append(f"Confidence Score: {output.get('confidence', 1.0):.4f}")
        return "\n".join(lines)

    if isinstance(output, dict) and output.get("task") == "language_identification":
        lines.append("-" * 70)
        lines.append("OUTPUT TYPE: Audio / Language Identification")
        lines.append("-" * 70)
        lines.append(
            f"Detected Language: {output.get('language', '')} "
            f"(Class ID: {output.get('class_id', 0)}, Score: {output.get('confidence', 0.0):.4f})"
        )
        lines.append(f"Audio Duration   : {output.get('duration_seconds', 0.0):.2f} s")
        top_preds = output.get("top_predictions", [])
        if top_preds:
            lines.append("\nTop Language Predictions:")
            for idx, p in enumerate(top_preds, start=1):
                lines.append(f"  #{idx} {p.get('language'):<12} Score: {p.get('score'):.4f} (ID: {p.get('class_id')})")
        return "\n".join(lines)

    # 3. Sequence Classification Output Dict
    if isinstance(output, dict) and "label" in output and ("score" in output or "probabilities" in output):
        lines.append("-" * 70)
        lines.append("OUTPUT TYPE: Sequence Classification")
        lines.append("-" * 70)
        label = output.get("label", "")
        score = output.get("score", 0.0)
        cid = output.get("class_id", 0)
        lines.append(f"Predicted Class: {label} (Class ID: {cid}, Confidence: {score:.4f})")
        probs = output.get("probabilities", {})
        if probs:
            lines.append("\nClass Probabilities:")
            for k, v in probs.items():
                lines.append(f"  {k}: {v:.4f}")
        return "\n".join(lines)

    # 4. Question Answering Output Dict
    if isinstance(output, dict) and "answer" in output and "score" in output:
        lines.append("-" * 70)
        lines.append("OUTPUT TYPE: Question Answering")
        lines.append("-" * 70)
        lines.append(f"Answer: \"{output.get('answer', '')}\"")
        lines.append(f"Confidence Score: {output.get('score', 0.0):.4f}")
        if "start" in output and "end" in output:
            lines.append(f"Token Span: [{output.get('start')}, {output.get('end')}]")
        return "\n".join(lines)

    # 5. Generic Dictionary or List Output
    if isinstance(output, (dict, list)):
        lines.append("-" * 70)
        lines.append(f"OUTPUT TYPE: Structured {type(output).__name__}")
        lines.append("-" * 70)
        try:
            lines.append(json.dumps(output, indent=2, ensure_ascii=False))
        except Exception:
            lines.append(str(output))
        return "\n".join(lines)

    # 6. Fallback Plain Output
    lines.append("-" * 70)
    lines.append(f"OUTPUT TYPE: {type(output).__name__}")
    lines.append("-" * 70)
    lines.append(str(output))
    return "\n".join(lines)


def run_model_test(
    model_path: Optional[str] = None,
    input_source: Optional[str] = None,
    input_file: Optional[str] = None,
    input_bucket: Optional[str] = None,
    input_key: Optional[str] = None,
    framework: str = "auto",
    device: str = "auto",
    timeout_seconds: float = 30.0,
    verbose: bool = False,
    print_output: bool = True,
    model_bucket: Optional[str] = None,
    model_key: Optional[str] = None,
    model_dir: Optional[str] = None,
    model_id: Optional[str] = None,
    task: str = "auto",
    input_type: str = "auto",
    triton_url: Optional[str] = None,
    **kwargs: Any,
) -> ModelTestReport:
    """
    Executes a complete manual model test workflow using the official ml_framework public API.

    Lifecycle stages:
    [STAGE 1/6] Model artifact retrieval
    [STAGE 2/6] Input artifact retrieval
    [STAGE 3/6] Model loader/backend selection
    [STAGE 4/6] Model loading & server readiness probe
    [STAGE 5/6] Inference execution
    [STAGE 6/6] Output validation

    Distinguishes PASS, PARTIAL, BLOCKED, and FAIL statuses.
    """
    overall_start = time.perf_counter()

    # Resolve model source from bucket/key/dir if model_path is not explicitly provided
    resolved_model_path = model_path
    if not resolved_model_path:
        bkt = model_bucket or kwargs.get("bucket")
        k = model_key or model_dir or kwargs.get("key") or kwargs.get("object_key")
        if bkt and k:
            resolved_model_path = f"s3://{str(bkt).strip()}/{str(k).lstrip('/')}"
        elif model_dir and os.path.exists(model_dir):
            resolved_model_path = model_dir

    report = ModelTestReport(
        success=False,
        model_path=resolved_model_path or "unspecified",
        status="FAIL",
        stage_reached="stage_1_model_retrieval",
        framework=framework,
    )

    manager = None
    temp_input_dir = None

    try:
        # =============================================================
        # STAGE 1/6: Model artifact retrieval
        # =============================================================
        report.stage_reached = "stage_1_model_retrieval"
        if print_output:
            print("=" * 80)
            print("ML FRAMEWORK — MANUAL MODEL TEST RUNNER")
            print("=" * 80)
            print("[STAGE 1/6] Model Artifact Retrieval")
            if model_bucket and (model_key or model_dir):
                print(f"  Storage Source    : MinIO Object Storage (s3://{model_bucket}/{model_key or model_dir})")
                print(f"  Bucket            : {model_bucket}")
                print(f"  Object/Prefix     : {model_key or model_dir}")
            else:
                print(f"  Storage Source    : Local / URI ({resolved_model_path})")
            print(f"  Target Model Path : {resolved_model_path}")
            print(f"  Task Contract     : {task}")
            print(f"  Input Type        : {input_type}")
            print(f"  Framework Setting : {framework}")
            print(f"  Device Setting    : {device}")
            print(f"  Timeout (seconds) : {timeout_seconds}")

        if not resolved_model_path or not str(resolved_model_path).strip():
            report.status = "BLOCKED"
            report.blocker_reason = "Model path cannot be empty (provide model_path or --bucket and --key/--model-dir)."
            report.error = report.blocker_reason
            report.error_type = "ValueError"
            report.exit_code = 2
            if print_output:
                print(f"  Status            : BLOCKED ({report.error})")
            return report

        model_path = resolved_model_path

        # Reject direct string input if provided via legacy or keyword argument
        if "input_text" in kwargs and kwargs["input_text"] is not None:
            report.status = "BLOCKED"
            report.blocker_reason = (
                "Direct string input is not supported. Please provide one of the supported input sources: "
                "1) Local file (--input-file or -i), 2) MinIO URI (s3:// or minio://), 3) Kafka (kafka://), or 4) MinIO input (--input-bucket and --input-key)."
            )
            report.error = report.blocker_reason
            report.error_type = "InvalidInputSourceError"
            report.exit_code = 1
            if print_output:
                print(f"\n[ERROR] {report.error}")
            return report

        # =============================================================
        # STAGE 2/6: Input artifact retrieval
        # =============================================================
        report.stage_reached = "stage_2_input_retrieval"
        if print_output:
            print("\n[STAGE 2/6] Input Artifact Retrieval")

        # Check if input is provided via input_bucket and input_key
        in_bkt = input_bucket or kwargs.get("input_bucket")
        in_key = input_key or kwargs.get("input_key") or kwargs.get("input_object_key")

        input_data = None
        if in_bkt and in_key:
            if print_output:
                print(f"  Retrieving from MinIO : s3://{in_bkt}/{in_key}")
            try:
                staged_input_path, temp_input_dir = download_input_artifact(
                    bucket=in_bkt,
                    key=in_key,
                )
                input_data = staged_input_path
                file_size = os.path.getsize(staged_input_path)
                ext = os.path.splitext(staged_input_path)[1]
                report.input_source_type = "minio_staged"
                report.input_data_summary = f"MinIO: s3://{in_bkt}/{in_key} ({file_size} bytes, ext='{ext}') -> {staged_input_path}"
                if print_output:
                    print(f"  Staging Path          : {staged_input_path} (size={file_size} bytes, ext='{ext}')")
                    print(f"  Input Download Status : SUCCESS")
            except Exception as dl_err:
                report.status = "BLOCKED"
                report.blocker_reason = f"Failed to retrieve input artifact from MinIO s3://{in_bkt}/{in_key}: {dl_err}"
                report.error = report.blocker_reason
                report.error_type = type(dl_err).__name__
                report.exit_code = 2
                if print_output:
                    print(f"  Input Download Status : BLOCKED ({report.error})")
                return report
        else:
            raw_source = input_source if input_source is not None else input_file
            if raw_source is None:
                report.status = "BLOCKED"
                report.blocker_reason = (
                    "Input source is required. Direct string input is not supported. "
                    "Please provide one of the supported input sources: "
                    "1) Local file (--input-file or -i), 2) MinIO (s3:// or minio://), 3) Kafka (kafka://), or 4) MinIO input (--input-bucket and --input-key)."
                )
                report.error = report.blocker_reason
                report.error_type = "ValueError"
                report.exit_code = 2
                if print_output:
                    print(f"  Input Source Status   : BLOCKED ({report.error})")
                return report

            clean_source = str(raw_source).strip()
            if not clean_source:
                report.status = "BLOCKED"
                report.blocker_reason = "Input source cannot be empty or whitespace only."
                report.error = report.blocker_reason
                report.error_type = "ValueError"
                report.exit_code = 1
                if print_output:
                    print(f"  Input Source Status   : BLOCKED ({report.error})")
                return report

            lower_source = clean_source.lower()
            if lower_source.startswith(("s3://", "minio://", "seaweedfs://")):
                report.input_source_type = "minio"
                report.input_data_summary = f"MinIO URI: {clean_source}"
                input_data = clean_source
                if print_output:
                    print(f"  Input Source          : MinIO URI ({clean_source})")

            elif lower_source.startswith("kafka://"):
                report.input_source_type = "kafka"
                report.input_data_summary = f"Kafka URI: {clean_source}"
                input_data = clean_source
                if print_output:
                    print(f"  Input Source          : Kafka URI ({clean_source})")

            elif lower_source.startswith("file://") or os.path.exists(os.path.abspath(os.path.expanduser(clean_source))):
                raw_path = clean_source[7:] if lower_source.startswith("file://") else clean_source
                resolved_path = os.path.abspath(os.path.expanduser(raw_path))
                if not os.path.exists(resolved_path):
                    report.status = "BLOCKED"
                    report.blocker_reason = f"Specified input file does not exist: '{resolved_path}'"
                    report.error = report.blocker_reason
                    report.error_type = "FileNotFoundError"
                    report.exit_code = 1
                    if print_output:
                        print(f"  Input Source Status   : BLOCKED ({report.error})")
                    return report
                report.input_source_type = "file"
                report.input_data_summary = f"File: {resolved_path}"
                input_data = resolved_path
                if print_output:
                    print(f"  Input Source          : Local File ({resolved_path})")

            elif (
                input_file is not None
                or lower_source.startswith(("/", "./", "../", "~/", "\\"))
                or any(lower_source.endswith(ext) for ext in [
                    ".txt", ".json", ".jsonl", ".csv", ".tsv", ".wav", ".mp3", ".flac", ".ogg", ".bin"
                ])
            ):
                raw_path = clean_source[7:] if lower_source.startswith("file://") else clean_source
                resolved_path = os.path.abspath(os.path.expanduser(raw_path))
                report.status = "BLOCKED"
                report.blocker_reason = f"Specified input file does not exist: '{resolved_path}'"
                report.error = report.blocker_reason
                report.error_type = "FileNotFoundError"
                report.exit_code = 1
                if print_output:
                    print(f"  Input Source Status   : BLOCKED ({report.error})")
                return report

            else:
                report.status = "BLOCKED"
                report.blocker_reason = (
                    f"Direct string input is not supported: '{clean_source[:50]}...'. "
                    "Please provide one of the supported input sources: "
                    "1) Local file (path or file://), 2) MinIO (s3:// or minio://), 3) Kafka (kafka://), or 4) MinIO input (--input-bucket and --input-key)."
                )
                report.error = report.blocker_reason
                report.error_type = "InvalidInputSourceError"
                report.exit_code = 1
                if print_output:
                    print(f"  Input Source Status   : BLOCKED ({report.error})")
                return report

        # =============================================================
        # STAGE 3/6: Model loader/backend selection
        # =============================================================
        report.stage_reached = "stage_3_backend_selection"
        if print_output:
            print("\n[STAGE 3/6] Model Loader / Backend Selection")

        is_remote_model = str(model_path).lower().startswith(("s3://", "minio://", "seaweedfs://"))
        if not is_remote_model and os.path.exists(os.path.abspath(os.path.expanduser(model_path))):
            try:
                detector = FormatDetector()
                clean_model_path = os.path.abspath(os.path.expanduser(model_path))
                detected_fw, detected_arch, detected_type = detector.detect(
                    clean_model_path, configured_framework=framework
                )
                report.detected_framework = detected_fw
                report.detected_architecture = detected_arch
                if print_output:
                    print(f"  Pre-detected Format   : framework='{detected_fw}', architecture='{detected_arch}', type='{detected_type}'")
            except Exception as det_err:
                if print_output:
                    print(f"  Pre-detected Format   : Unable to pre-detect ({det_err})")
        else:
            if print_output:
                print(f"  Remote Storage Source : Detection will occur during staging from MinIO")

        triton_dict = {}
        if triton_url:
            triton_dict["endpoint_url"] = triton_url
        if model_id:
            triton_dict["model_name"] = model_id

        target_backend = triton_dict.get("backend", "python")
        if print_output:
            print(f"  Target Inference Engine: NVIDIA Triton Inference Server")
            print(f"  Target Triton Backend  : {target_backend}")
            print(f"  Triton Model Name      : {model_id or 'auto-resolved'}")

        # =============================================================
        # STAGE 4/6: Model loading & server readiness probe
        # =============================================================
        report.stage_reached = "stage_4_model_loading"
        if print_output:
            print("\n[STAGE 4/6] Model Loading & Server Readiness Probe")
            print("  Loading model via ml.load()...")

        load_start = time.perf_counter()
        load_kwargs: Dict[str, Any] = {
            "model_source": model_path,
            "framework": framework,
            "device": device,
            "timeout_seconds": timeout_seconds,
        }
        if model_bucket is not None:
            load_kwargs["model_bucket"] = model_bucket
        if model_key is not None:
            load_kwargs["model_key"] = model_key
        if model_dir is not None:
            load_kwargs["model_dir"] = model_dir
        if model_id is not None:
            load_kwargs["model_id"] = model_id
        if task is not None and task != "auto":
            load_kwargs["task"] = task
        if input_type is not None and input_type != "auto":
            load_kwargs["input_type"] = input_type
        if triton_dict:
            load_kwargs["triton_config"] = triton_dict

        manager = ml.load(**load_kwargs)
        report.load_duration_ms = (time.perf_counter() - load_start) * 1000

        # Inspect active model adapter and state
        report.model_state = manager.state.value if hasattr(manager.state, "value") else str(manager.state)
        active_adapter = getattr(manager, "_active_model_instance", None)
        if active_adapter is not None:
            report.adapter_class = type(active_adapter).__name__
            if hasattr(active_adapter, "architecture") and active_adapter.architecture:
                report.detected_architecture = active_adapter.architecture
            if not report.detected_framework:
                if hasattr(active_adapter, "framework") and active_adapter.framework:
                    report.detected_framework = active_adapter.framework
                elif "Transformers" in report.adapter_class:
                    report.detected_framework = "transformers"
                elif "Spacy" in report.adapter_class:
                    report.detected_framework = "spacy"
                elif "ONNX" in report.adapter_class:
                    report.detected_framework = "onnx"
                elif "Speech" in report.adapter_class:
                    report.detected_framework = "speech"

        if print_output:
            print(f"  Model Load Status     : SUCCESS ({report.load_duration_ms:.2f} ms)")
            print(f"  Active Model Adapter  : {report.adapter_class or 'TritonInferenceEngine'}")
            print(f"  Manager State         : {report.model_state}")
            print(f"  Server / Model Ready  : {manager.is_ready}")

        if not manager.is_ready:
            raise StateError(f"ExecutionManager is not in READY state (current: {manager.state})")

        # =============================================================
        # STAGE 5/6: Inference execution
        # =============================================================
        report.stage_reached = "stage_5_inference"
        if print_output:
            print("\n[STAGE 5/6] Inference Execution")
            print("  Executing inference via manager.process()...")

        infer_start = time.perf_counter()
        exec_result = manager.process(input_data)
        report.inference_duration_ms = (time.perf_counter() - infer_start) * 1000

        if not exec_result.is_success():
            report.status = "FAIL"
            report.error = exec_result.error or "Model inference reported failure."
            report.error_type = exec_result.error_type or "ExecutionError"
            report.exit_code = 1
            if print_output:
                print(f"  Inference Status      : FAILED ({report.error})")
            return report

        if print_output:
            print(f"  Inference Status      : SUCCESS ({report.inference_duration_ms:.2f} ms)")

        # =============================================================
        # STAGE 6/6: Output validation
        # =============================================================
        report.stage_reached = "stage_6_output_validation"
        if print_output:
            print("\n[STAGE 6/6] Output Validation")

        report.raw_output = exec_result.output
        report.formatted_output = format_model_output(report.raw_output, verbose=verbose)

        # Extract entities and tokens for structured summary
        if hasattr(report.raw_output, "ents"):
            report.entities = [
                {
                    "entity": ent.label_,
                    "word": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char,
                }
                for ent in report.raw_output.ents
            ]
            report.tokens = [{"token": t.text} for t in report.raw_output]
        elif isinstance(report.raw_output, dict):
            if "entities" in report.raw_output:
                report.entities = report.raw_output.get("entities", [])
            if "tokens" in report.raw_output:
                report.tokens = report.raw_output.get("tokens", [])

        # Validate contract satisfaction
        is_contract_valid = False
        contract_notes = ""
        effective_task = task.lower()

        if effective_task == "transcription":
            if isinstance(report.raw_output, dict) and any(k in report.raw_output for k in ("transcription", "text")):
                transcript_txt = str(report.raw_output.get("transcription") or report.raw_output.get("text") or "").strip()
                if transcript_txt:
                    is_contract_valid = True
                    contract_notes = f"Transcription text received ({len(transcript_txt)} characters)"
                else:
                    contract_notes = "Transcription output was empty string"
            else:
                contract_notes = f"Expected dict containing 'transcription' or 'text', got {type(report.raw_output).__name__}"

        elif effective_task == "language_detection":
            if isinstance(report.raw_output, dict) and any(k in report.raw_output for k in ("language", "label")):
                lang_val = report.raw_output.get("language") or report.raw_output.get("label")
                is_contract_valid = True
                contract_notes = f"Language label detected: {lang_val}"
            else:
                contract_notes = f"Expected dict containing 'language' or 'label', got {type(report.raw_output).__name__}"

        elif effective_task == "denoising":
            if isinstance(report.raw_output, dict) and "enhanced_audio" in report.raw_output:
                is_contract_valid = True
                contract_notes = "Enhanced audio payload present"
            else:
                contract_notes = f"Expected dict containing 'enhanced_audio', got {type(report.raw_output).__name__}"

        else:
            if report.raw_output is not None:
                is_contract_valid = True
                contract_notes = f"Valid output received ({type(report.raw_output).__name__})"

        if is_contract_valid:
            report.status = "PASS"
            report.success = True
            report.exit_code = 0
            if print_output:
                print(f"  Contract Validation   : PASS ({contract_notes})")
                print("\n" + report.formatted_output)
        else:
            report.status = "FAIL"
            report.success = False
            report.error = f"Output contract validation failed: {contract_notes}"
            report.error_type = "ContractValidationError"
            report.exit_code = 1
            if print_output:
                print(f"  Contract Validation   : FAIL ({contract_notes})")

    except DependencyMissingError as dep_err:
        report.status = "PARTIAL"
        report.blocker_reason = f"Missing container dependency: {dep_err.message}"
        report.error = dep_err.message
        report.error_type = "DependencyMissingError"
        report.exit_code = 1
        if print_output:
            print(f"\n[PARTIAL] Model retrieved, but Triton container lacks required dependency:")
            print(f"  {dep_err.message}")

    except ModelFormatError as fmt_err:
        if "incompatible" in str(fmt_err).lower() or "backend" in str(fmt_err).lower():
            report.status = "PARTIAL"
            report.blocker_reason = f"Backend incompatibility: {fmt_err.message}"
            report.exit_code = 1
        else:
            report.status = "BLOCKED"
            report.blocker_reason = f"Model format error: {fmt_err.message}"
            report.exit_code = 2
        report.error = fmt_err.message
        report.error_type = "ModelFormatError"
        if print_output:
            print(f"\n[{report.status}] {report.error}")

    except (InvalidInputSourceError, InputProcessingError, ConfigurationError) as block_err:
        report.status = "BLOCKED"
        report.blocker_reason = block_err.message
        report.error = block_err.message
        report.error_type = type(block_err).__name__
        report.exit_code = 2 if isinstance(block_err, (ModelLoadError, ConfigurationError)) else 1
        if print_output:
            print(f"\n[BLOCKED] {report.error_type}: {report.error}")

    except MLFrameworkError as fw_err:
        report.status = "BLOCKED" if isinstance(fw_err, (ModelLoadError, ConfigurationError)) else "FAIL"
        report.error = fw_err.message
        report.error_type = type(fw_err).__name__
        report.blocker_reason = fw_err.message
        report.exit_code = 2 if isinstance(fw_err, (ModelLoadError, EnvironmentError, ConfigurationError)) else 1
        if print_output:
            print(f"\n[{report.status}] {report.error_type}: {report.error}")
            if fw_err.details:
                print(f"Details: {fw_err.details}")

    except Exception as exc:
        report.status = "FAIL"
        report.error = str(exc)
        report.error_type = type(exc).__name__
        report.exit_code = 1
        if print_output:
            print(f"\n[FAIL] {report.error_type}: {report.error}")

    finally:
        # Clean up temporary staged input files
        if temp_input_dir and os.path.exists(temp_input_dir):
            try:
                shutil.rmtree(temp_input_dir, ignore_errors=True)
            except Exception:
                pass

        # Model cleanup and unload lifecycle orchestration
        if manager is not None:
            if print_output:
                print("\nInitiating model cleanup and unload...")
            try:
                manager.unload()
                report.cleanup_successful = True
                if print_output:
                    print(f"Model Unload          : SUCCESS (State: {manager.state})")
            except Exception as unload_err:
                logger.warning(f"Error during manager.unload(): {unload_err}")
                if print_output:
                    print(f"[WARNING] Error during unload: {unload_err}")

            try:
                manager.stop()
                if print_output:
                    print(f"Manager Stop          : SUCCESS (State: {manager.state})")
            except Exception as stop_err:
                logger.warning(f"Error during manager.stop(): {stop_err}")
                if print_output:
                    print(f"[WARNING] Error during stop: {stop_err}")

        report.total_duration_ms = (time.perf_counter() - overall_start) * 1000

        if print_output:
            print("\n" + "=" * 80)
            print(f"MANUAL TEST RUN RESULT: {report.status} (Stage Reached: {report.stage_reached}, Total Time: {report.total_duration_ms:.2f} ms, Exit Code: {report.exit_code})")
            if report.blocker_reason:
                print(f"Blocker / Note        : {report.blocker_reason}")
            print("=" * 80)

    return report


def build_cli_parser() -> argparse.ArgumentParser:
    """Builds the command-line argument parser for the manual test runner."""
    parser = argparse.ArgumentParser(
        description="ML Framework Manual Model Test Runner — Load, execute, and inspect models from local or MinIO storage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model_path_pos",
        nargs="?",
        default=None,
        help="Local filesystem path or storage URI (s3://, minio://) to model directory or artifact file (positional).",
    )
    parser.add_argument(
        "-m", "--model-path",
        dest="model_path_opt",
        default=None,
        help="Local filesystem path or storage URI to model directory or artifact file (flag option).",
    )
    parser.add_argument(
        "-b", "--bucket",
        default=os.environ.get("MINIO_MODEL_BUCKET") or os.environ.get("MINIO_BUCKET"),
        help="MinIO / S3 storage bucket name for model artifacts.",
    )
    parser.add_argument(
        "-k", "--key", "--object-key",
        dest="model_key",
        default=None,
        help="MinIO / S3 object key or directory prefix for model.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Model directory prefix in MinIO or local filesystem path.",
    )
    parser.add_argument(
        "--model-id", "--model-name",
        dest="model_id",
        default=None,
        help="Model identifier / Triton model name.",
    )
    parser.add_argument(
        "--task",
        default="auto",
        choices=["auto", "transcription", "language_detection", "denoising"],
        help="Inference task to execute ('transcription', 'language_detection', 'denoising').",
    )
    parser.add_argument(
        "--input-type",
        default="auto",
        choices=["auto", "audio", "text"],
        help="Expected input type contract ('audio', 'text').",
    )
    parser.add_argument(
        "--triton-url",
        default=os.environ.get("TRITON_SERVER_URL", "http://localhost:8010"),
        help="NVIDIA Triton Inference Server HTTP endpoint URL.",
    )
    parser.add_argument(
        "-f", "--framework",
        default="auto",
        choices=["auto", "spacy", "transformers", "huggingface", "onnx", "speech", "nemo", "ctranslate2", "deepfilternet", "speechbrain"],
        help="Explicit model framework override (defaults to auto-detection).",
    )
    parser.add_argument(
        "-i", "--input-file", "--input",
        dest="input_source",
        default=None,
        help="Path to local file (or file://), MinIO URI (s3://, minio://), or Kafka URI (kafka://) containing input payload to ingest and process. Direct string input is not supported.",
    )
    parser.add_argument(
        "--input-bucket",
        default=None,
        help="MinIO / S3 storage bucket name containing input artifact.",
    )
    parser.add_argument(
        "--input-key", "--input-object-key",
        dest="input_key",
        default=None,
        help="MinIO / S3 object key for input artifact (e.g. 'audio /chunk_000.mp3').",
    )
    parser.add_argument(
        "-d", "--device",
        default="auto",
        help="Hardware compute device preference ('auto', 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Maximum execution timeout in seconds.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Display detailed token predictions, confidence metrics, and stage timestamps.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print structured test report as JSON to standard output.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for running manual model tests."""
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    # Resolve model path from positional, flag, or bucket/key arguments
    model_path = args.model_path_opt or args.model_path_pos
    if not model_path:
        key = args.model_key or args.model_dir
        if args.bucket and key:
            model_path = f"s3://{str(args.bucket).strip()}/{str(key).lstrip('/')}"
        elif args.model_dir and os.path.exists(args.model_dir):
            model_path = args.model_dir

    if not model_path and not (args.bucket and (args.model_key or args.model_dir)):
        parser.print_usage(sys.stderr)
        print("Error: model path is required (provide as positional argument, via -m/--model-path, or via --bucket and --key/--model-dir).", file=sys.stderr)
        return 2

    has_input = bool(args.input_source or (args.input_bucket and args.input_key))
    if not has_input:
        parser.print_usage(sys.stderr)
        print("Error: input source is required. Direct string input is not supported. Please provide a local file (-i/--input-file), MinIO URI (s3://, minio://), or --input-bucket and --input-key.", file=sys.stderr)
        return 2

    # In JSON mode, suppress intermediate console printing
    print_output = not args.json

    report = run_model_test(
        model_path=model_path,
        input_source=args.input_source,
        input_bucket=args.input_bucket,
        input_key=args.input_key,
        framework=args.framework,
        device=args.device,
        timeout_seconds=args.timeout,
        verbose=args.verbose,
        print_output=print_output,
        model_bucket=args.bucket,
        model_key=args.model_key,
        model_dir=args.model_dir,
        model_id=args.model_id,
        task=args.task,
        input_type=args.input_type,
        triton_url=args.triton_url,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))

    return report.exit_code


if __name__ == "__main__":
    sys.exit(main())
