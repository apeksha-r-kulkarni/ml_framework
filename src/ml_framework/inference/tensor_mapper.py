"""
TensorMapper for translating between ML Framework domain objects and Triton KServe v2 tensors.

Decouples model-specific tensor structures from the generic execution engine.
Translates preprocessed inputs (audio signals, text strings, raw dictionaries, numpy arrays)
into typed Triton request payloads and decodes Triton inference responses into standardized outputs.
"""

import io
import wave
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from ..errors.exceptions import InferenceError, InputProcessingError, DependencyMissingError

logger = logging.getLogger("ml_framework.inference.tensor_mapper")


class TensorMapper:
    """
    Translates between Python domain inputs/outputs and Triton KServe v2 tensor representations.
    """

    def __init__(
        self,
        input_tensor_map: Optional[Dict[str, str]] = None,
        output_tensor_map: Optional[Dict[str, str]] = None,
        default_sample_rate: int = 16000,
        task: str = "auto",
    ):
        self.input_tensor_map = dict(input_tensor_map or {})
        self.output_tensor_map = dict(output_tensor_map or {})
        self.default_sample_rate = default_sample_rate
        self.task = task

    @staticmethod
    def _numpy_dtype_to_triton(dtype: Any) -> str:
        """Maps NumPy dtypes to Triton KServe v2 datatype strings."""
        if np is None:
            raise DependencyMissingError(
                "numpy is required for tensor dtype mapping. Install via: pip install numpy",
                missing_packages=["numpy"],
            )
        if np.issubdtype(dtype, np.floating):
            if dtype == np.float16:
                return "FP16"
            elif dtype == np.float64:
                return "FP64"
            return "FP32"
        elif np.issubdtype(dtype, np.signedinteger):
            if dtype == np.int8:
                return "INT8"
            elif dtype == np.int16:
                return "INT16"
            elif dtype == np.int64:
                return "INT64"
            return "INT32"
        elif np.issubdtype(dtype, np.unsignedinteger):
            if dtype == np.uint8:
                return "UINT8"
            elif dtype == np.uint16:
                return "UINT16"
            elif dtype == np.uint32:
                return "UINT32"
            return "UINT64"
        elif np.issubdtype(dtype, np.bool_):
            return "BOOL"
        elif np.issubdtype(dtype, np.flexible) or np.issubdtype(dtype, np.object_):
            return "BYTES"
        return "FP32"

    def _decode_audio_input(self, audio_data: Any) -> Tuple[Any, int]:
        """
        Extracts 1D float32 audio samples and sample rate from arbitrary audio representations.
        """
        # 1. Check if input is already a NumPy array
        if isinstance(audio_data, np.ndarray):
            flat = audio_data.astype(np.float32).flatten()
            if flat.size == 0:
                raise InferenceError("Audio input array is empty (0 samples).")
            return flat, self.default_sample_rate

        # 2. Check if input is a list or sequence of numbers
        if isinstance(audio_data, (list, tuple)) and audio_data and isinstance(audio_data[0], (int, float)):
            flat = np.array(audio_data, dtype=np.float32).flatten()
            if flat.size == 0:
                raise InferenceError("Audio input array is empty (0 samples).")
            return flat, self.default_sample_rate

        # 3. Check if input is tuple (signal, sample_rate)
        if isinstance(audio_data, tuple) and len(audio_data) == 2:
            sig, sr = audio_data
            flat_sig, _ = self._decode_audio_input(sig)
            return flat_sig, int(sr)

        # 4. Check if input is dict containing audio / waveform / signal
        if isinstance(audio_data, dict):
            sr = int(
                audio_data.get("sample_rate")
                or audio_data.get("sampling_rate")
                or self.default_sample_rate
            )
            for k in ("audio_signal", "AUDIO_SIGNAL", "audio", "waveform", "signal", "raw_audio"):
                if k in audio_data:
                    flat_sig, _ = self._decode_audio_input(audio_data[k])
                    return flat_sig, sr
            if "bytes" in audio_data:
                return self._decode_audio_bytes(audio_data["bytes"])
            if "path" in audio_data:
                with open(audio_data["path"], "rb") as f:
                    return self._decode_audio_bytes(f.read())

        # 5. Check if input is string path to an existing audio file
        if isinstance(audio_data, str) and (
            audio_data.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".bin"))
            or (len(audio_data) < 512 and ("/" in audio_data or "\\" in audio_data))
        ):
            clean_p = audio_data[7:] if audio_data.startswith("file://") else audio_data
            import os
            if os.path.isfile(clean_p):
                with open(clean_p, "rb") as f:
                    return self._decode_audio_bytes(f.read())

        # 6. Check if input is binary bytes (WAV, etc.)
        if isinstance(audio_data, (bytes, bytearray, memoryview)):
            return self._decode_audio_bytes(bytes(audio_data))

        raise InferenceError(
            f"Unsupported audio input type for Triton speech inference: {type(audio_data).__name__}"
        )

    def _decode_audio_bytes(self, raw_bytes: bytes) -> Tuple[Any, int]:
        """Decodes raw audio bytes into float32 samples and sample rate."""
        if not raw_bytes:
            raise InferenceError("Audio binary payload is empty (0 bytes).")

        # Try using soundfile if available
        try:
            import soundfile as sf
            data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            return data.flatten(), int(sr)
        except Exception:
            pass

        # Try using framework audio utility
        try:
            from ..input.audio import decode_and_resample_audio
            data, duration = decode_and_resample_audio(raw_bytes, target_sampling_rate=self.default_sample_rate)
            return data.flatten(), self.default_sample_rate
        except Exception:
            pass

        # Fallback to standard library wave for WAV PCM
        if raw_bytes.startswith(b"RIFF"):
            try:
                with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
                    sr = wf.getframerate()
                    sw = wf.getsampwidth()
                    ch = wf.getnchannels()
                    frames = wf.readframes(wf.getnframes())
                    if sw == 1:
                        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
                    elif sw == 2:
                        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    elif sw == 4:
                        data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
                    else:
                        raise InferenceError(f"Unsupported WAV sample width: {sw} bytes")
                    if ch > 1:
                        data = data.reshape(-1, ch).mean(axis=1)
                    return data.flatten(), int(sr)
            except Exception as e:
                raise InferenceError(f"Failed to parse WAV audio bytes: {e}") from e

        raise InferenceError("Unable to decode audio bytes: soundfile is not available and payload is not standard PCM WAV.")

    def build_request(
        self,
        preprocessed_input: Any,
        model_name: Optional[str] = None,
        model_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, str]]]]:
        """
        Constructs the inputs and requested outputs structures for Triton KServe v2 REST request
        adhering strictly to the model's task contract.

        Returns:
            Tuple of (inputs_list, outputs_list)
        """
        # Resolve task from model_metadata or instance config
        meta_task = (model_metadata.get("task") if isinstance(model_metadata, dict) else None) or self.task or "auto"
        norm_task = meta_task.lower().strip()

        # Case A: Input is already a fully formed list of Triton input dicts
        if (
            isinstance(preprocessed_input, list)
            and preprocessed_input
            and isinstance(preprocessed_input[0], dict)
            and "name" in preprocessed_input[0]
            and "data" in preprocessed_input[0]
        ):
            return preprocessed_input, None

        # Case B: Direct dictionary with exact Triton input keys (e.g. {"AUDIO_SIGNAL": ..., "SAMPLE_RATE": ...})
        if isinstance(preprocessed_input, dict) and any(k in preprocessed_input for k in ("AUDIO_SIGNAL", "NOISY_AUDIO")):
            sig_key = "NOISY_AUDIO" if "NOISY_AUDIO" in preprocessed_input else "AUDIO_SIGNAL"
            sig = preprocessed_input[sig_key]
            if isinstance(sig, np.ndarray):
                sig_data = sig.astype(np.float32).flatten().tolist()
            elif isinstance(sig, list):
                sig_data = [float(x) for x in sig]
            else:
                flat_sig, _ = self._decode_audio_input(sig)
                sig_data = flat_sig.tolist()

            sr_val = preprocessed_input.get("SAMPLE_RATE", self.default_sample_rate)
            sr_item = int(sr_val.item()) if isinstance(sr_val, np.ndarray) else int(sr_val[0]) if isinstance(sr_val, list) else int(sr_val)

            inputs = [
                {
                    "name": sig_key,
                    "shape": [len(sig_data)],
                    "datatype": "FP32",
                    "data": sig_data,
                },
                {
                    "name": "SAMPLE_RATE",
                    "shape": [1],
                    "datatype": "INT32",
                    "data": [sr_item],
                },
            ]
            if norm_task == "denoising":
                outputs = [{"name": self.output_tensor_map.get("enhanced_audio", "ENHANCED_AUDIO")}]
            elif norm_task == "language_detection":
                outputs = [{"name": self.output_tensor_map.get("language", "LANGUAGE")}]
            else:
                outputs = [{"name": self.output_tensor_map.get("transcript", "TRANSCRIPT")}]
            return inputs, outputs

        # Task contract input validation:
        # Denoising and Transcription require audio input, reject pure text string
        import os
        is_existing_path = isinstance(preprocessed_input, str) and (
            os.path.exists(preprocessed_input)
            or (preprocessed_input.startswith("file://") and os.path.exists(preprocessed_input[7:]))
        )
        if isinstance(preprocessed_input, str) and not is_existing_path:
            if norm_task == "denoising":
                raise InputProcessingError(
                    f"Denoising task requires audio input (audio samples, bytes, or audio file), received text: '{preprocessed_input[:40]}...'"
                )
            if norm_task == "transcription":
                raise InputProcessingError(
                    f"Transcription task requires audio input (audio samples, bytes, or audio file), received text: '{preprocessed_input[:40]}...'"
                )

        # Case C: Denoising task
        if norm_task == "denoising":
            audio_samples, sample_rate = self._decode_audio_input(preprocessed_input)
            if not np.all(np.isfinite(audio_samples)):
                raise InferenceError("Input audio contains non-finite values (NaN or Inf).")

            audio_signal_name = self.input_tensor_map.get("audio", "NOISY_AUDIO" if "NOISY_AUDIO" in self.input_tensor_map else "AUDIO_SIGNAL")
            sample_rate_name = self.input_tensor_map.get("sample_rate", "SAMPLE_RATE")
            inputs = [
                {
                    "name": audio_signal_name,
                    "shape": [len(audio_samples)],
                    "datatype": "FP32",
                    "data": audio_samples.tolist(),
                },
                {
                    "name": sample_rate_name,
                    "shape": [1],
                    "datatype": "INT32",
                    "data": [sample_rate],
                },
            ]
            output_name = self.output_tensor_map.get("enhanced_audio", "ENHANCED_AUDIO")
            outputs = [{"name": output_name}]
            return inputs, outputs

        # Case D: Language detection task
        if norm_task == "language_detection":
            if isinstance(preprocessed_input, str) and not is_existing_path:
                text_input_name = self.input_tensor_map.get("text", "TEXT")
                inputs = [
                    {
                        "name": text_input_name,
                        "shape": [1],
                        "datatype": "BYTES",
                        "data": [preprocessed_input],
                    }
                ]
            else:
                audio_samples, sample_rate = self._decode_audio_input(preprocessed_input)
                if not np.all(np.isfinite(audio_samples)):
                    raise InferenceError("Input audio contains non-finite values (NaN or Inf).")
                audio_signal_name = self.input_tensor_map.get("audio", "AUDIO_SIGNAL")
                sample_rate_name = self.input_tensor_map.get("sample_rate", "SAMPLE_RATE")
                inputs = [
                    {
                        "name": audio_signal_name,
                        "shape": [len(audio_samples)],
                        "datatype": "FP32",
                        "data": audio_samples.tolist(),
                    },
                    {
                        "name": sample_rate_name,
                        "shape": [1],
                        "datatype": "INT32",
                        "data": [sample_rate],
                    },
                ]
            output_name = self.output_tensor_map.get("language", "LANGUAGE")
            outputs = [{"name": output_name}]
            return inputs, outputs

        # Case E: Speech audio input (transcription or default speech)
        is_speech = (
            norm_task == "transcription"
            or isinstance(preprocessed_input, (bytes, bytearray, memoryview))
            or (isinstance(preprocessed_input, np.ndarray) and preprocessed_input.ndim <= 2)
            or (isinstance(preprocessed_input, tuple) and len(preprocessed_input) == 2)
            or (isinstance(preprocessed_input, dict) and any(
                k in preprocessed_input for k in ("audio", "waveform", "audio_signal", "signal", "bytes", "path")
            ))
            or is_existing_path
            or (model_name and any(k in model_name.lower() for k in ("speech", "conformer", "asr", "nemo", "whisper", "wav2vec")))
        )

        if is_speech:
            audio_samples, sample_rate = self._decode_audio_input(preprocessed_input)
            if not np.all(np.isfinite(audio_samples)):
                raise InferenceError("Input audio contains non-finite values (NaN or Inf).")

            audio_signal_name = self.input_tensor_map.get("audio", "AUDIO_SIGNAL")
            sample_rate_name = self.input_tensor_map.get("sample_rate", "SAMPLE_RATE")

            inputs = [
                {
                    "name": audio_signal_name,
                    "shape": [len(audio_samples)],
                    "datatype": "FP32",
                    "data": audio_samples.tolist(),
                },
                {
                    "name": sample_rate_name,
                    "shape": [1],
                    "datatype": "INT32",
                    "data": [sample_rate],
                },
            ]
            output_name = self.output_tensor_map.get("transcript", "TRANSCRIPT")
            outputs = [{"name": output_name}]
            return inputs, outputs

        # Case F: Text input (string) for non-audio models
        if isinstance(preprocessed_input, str):
            input_name = self.input_tensor_map.get("text", "TEXT")
            inputs = [
                {
                    "name": input_name,
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [preprocessed_input],
                }
            ]
            outputs = None
            return inputs, outputs

        # Case G: Generic dictionary mapping tensor names to values
        if isinstance(preprocessed_input, dict):
            inputs = []
            for raw_k, v in preprocessed_input.items():
                tensor_name = self.input_tensor_map.get(raw_k, raw_k)
                if isinstance(v, np.ndarray):
                    dtype_str = self._numpy_dtype_to_triton(v.dtype)
                    shape = list(v.shape)
                    data_list = v.tolist()
                elif isinstance(v, list):
                    dtype_str = "FP32" if (v and isinstance(v[0], float)) else "INT32" if (v and isinstance(v[0], int)) else "BYTES"
                    shape = [len(v)]
                    data_list = v
                elif isinstance(v, (int, float, str, bool)):
                    dtype_str = "FP32" if isinstance(v, float) else "INT32" if isinstance(v, int) else "BOOL" if isinstance(v, bool) else "BYTES"
                    shape = [1]
                    data_list = [v]
                else:
                    raise InferenceError(f"Unsupported data type for tensor '{tensor_name}': {type(v).__name__}")

                inputs.append({
                    "name": tensor_name,
                    "shape": shape,
                    "datatype": dtype_str,
                    "data": data_list,
                })
            return inputs, None

        raise InferenceError(
            f"Unable to map input of type {type(preprocessed_input).__name__} to Triton tensors."
        )

    def parse_response(
        self,
        response_data: Dict[str, Any],
        task: str = "auto",
    ) -> Any:
        """
        Decodes a Triton KServe v2 response dictionary into standard ML Framework output format
        strictly matching the task contract.

        Contracts:
        - transcription: {"task": "transcription", "text": str, "transcription": str}
        - language_detection: {"task": "language_detection", "language": str, "label": str, "confidence": float}
        - denoising: {"task": "denoising", "enhanced_audio": np.ndarray, "sample_rate": int}
        """
        if not isinstance(response_data, dict):
            raise InferenceError(f"Expected dict response from Triton, got {type(response_data).__name__}")

        raw_outputs = response_data.get("outputs")
        if not raw_outputs or not isinstance(raw_outputs, list):
            if "error" in response_data:
                raise InferenceError(f"Triton reported error: {response_data['error']}")
            return response_data

        parsed_tensors: Dict[str, Any] = {}
        for out_tensor in raw_outputs:
            t_name = out_tensor.get("name", "")
            t_data = out_tensor.get("data", [])
            mapped_name = self.output_tensor_map.get(t_name, t_name)
            parsed_tensors[mapped_name] = t_data

        # Resolve effective task
        norm_task = (task if task != "auto" else self.task if self.task != "auto" else "auto").lower().strip()
        if norm_task == "auto":
            if any(k in parsed_tensors for k in ("ENHANCED_AUDIO", "CLEAN_AUDIO")):
                norm_task = "denoising"
            elif any(k in parsed_tensors for k in ("LANGUAGE", "LABEL", "LANG", "PREDICTION")):
                norm_task = "language_detection"
            elif "TRANSCRIPT" in parsed_tensors:
                norm_task = "transcription"

        # 1. Denoising Contract
        if norm_task == "denoising" or any(k in parsed_tensors for k in ("ENHANCED_AUDIO", "CLEAN_AUDIO")):
            audio_raw = parsed_tensors.get(
                "ENHANCED_AUDIO",
                parsed_tensors.get("CLEAN_AUDIO", next(iter(parsed_tensors.values())))
            )
            sr_raw = parsed_tensors.get("SAMPLE_RATE", [self.default_sample_rate])
            sr = int(sr_raw[0] if isinstance(sr_raw, list) and sr_raw else sr_raw)
            if isinstance(audio_raw, np.ndarray):
                arr = audio_raw.astype(np.float32)
            elif isinstance(audio_raw, list):
                arr = np.array(audio_raw, dtype=np.float32)
            else:
                arr = np.array([audio_raw], dtype=np.float32)

            return {
                "task": "denoising",
                "enhanced_audio": arr,
                "sample_rate": sr,
            }

        # 2. Language Detection Contract
        if norm_task == "language_detection" or any(k in parsed_tensors for k in ("LANGUAGE", "LABEL", "LANG")):
            lang_raw = parsed_tensors.get(
                "LANGUAGE",
                parsed_tensors.get("LABEL", parsed_tensors.get("LANG", next(iter(parsed_tensors.values()))))
            )
            lang_str = str(lang_raw[0]) if isinstance(lang_raw, list) and len(lang_raw) > 0 else str(lang_raw)
            conf = 1.0
            if "CONFIDENCE" in parsed_tensors:
                c_val = parsed_tensors["CONFIDENCE"]
                conf = float(c_val[0] if isinstance(c_val, list) and c_val else c_val)
            elif "PROBABILITIES" in parsed_tensors:
                p_val = parsed_tensors["PROBABILITIES"]
                conf = float(max(p_val) if isinstance(p_val, list) and p_val else p_val)

            return {
                "task": "language_detection",
                "language": lang_str,
                "label": lang_str,
                "confidence": conf,
            }

        # 3. Transcription Contract
        if norm_task == "transcription" or "TRANSCRIPT" in parsed_tensors:
            transcript_raw = parsed_tensors.get("TRANSCRIPT", next(iter(parsed_tensors.values())) if len(parsed_tensors) == 1 else "")
            if isinstance(transcript_raw, list) and len(transcript_raw) > 0:
                transcript_str = str(transcript_raw[0])
            else:
                transcript_str = str(transcript_raw)

            return {
                "task": "transcription",
                "text": transcript_str,
                "transcription": transcript_str,
            }

        # 4. Single Output Tensor Unwrap
        if len(parsed_tensors) == 1:
            only_key = next(iter(parsed_tensors))
            val = parsed_tensors[only_key]
            if isinstance(val, list) and len(val) == 1:
                return val[0]
            return val

        return parsed_tensors
