"""
Audio decoding, normalization, and resampling utilities for the ML Framework.

Provides audio validation and conversion from arbitrary audio containers (WAV, MP3, FLAC, OGG)
or raw bytes into normalized 16kHz mono float32 waveforms for speech model consumption.
"""

import io
import os
from typing import Any, Tuple, Union, Optional
try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from ..errors.exceptions import (
    InputProcessingError,
    DependencyMissingError,
)

# Recognized audio magic headers
AUDIO_MAGIC_HEADERS = {
    b"RIFF": "wav",
    b"ID3": "mp3",
    b"fLaC": "flac",
    b"OggS": "ogg",
}


def is_valid_audio_payload(data: bytes) -> Tuple[bool, str]:
    """
    Validates whether binary data matches a known audio container magic signature.
    Supports WAV, MP3 (ID3v2 and raw MPEG sync headers), FLAC, and OGG.
    """
    if not data or len(data) < 4:
        return False, "empty_or_too_short"

    if data.startswith(b"ID3"):
        return True, "mp3"

    if data.startswith(b"RIFF"):
        return True, "wav"

    if data.startswith(b"fLaC"):
        return True, "flac"

    if data.startswith(b"OggS"):
        return True, "ogg"

    # MP3 without ID3 header: MPEG frame sync (11 consecutive 1 bits: 0xFF followed by 0xE0..0xFF)
    if data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return True, "mp3"

    return False, "unknown"


def decode_and_resample_audio(
    audio_input: Any,
    target_sampling_rate: int = 16000,
) -> Tuple[Any, float]:
    """
    Decodes audio input into a 1D float32 mono NumPy waveform at the target sampling rate.

    Supported input types:
    - Raw audio bytes / bytearray (WAV, MP3, FLAC, OGG)
    - Filesystem path to an audio file
    - Pre-decoded NumPy ndarray or float sequence
    - Dict with {"audio": ..., "sampling_rate": ...}

    Returns:
        Tuple of (waveform_1d_float32, duration_seconds)

    Raises:
        InputProcessingError: If input is empty, malformed, or an unsupported binary payload.
        DependencyMissingError: If soundfile or librosa/scipy are missing.
    """
    if np is None:
        raise DependencyMissingError(
            "numpy is required for audio decoding and resampling. "
            "Install via: pip install '.[speech]' or pip install numpy",
            missing_packages=["numpy"],
        )

    if audio_input is None:
        raise InputProcessingError("Audio input cannot be None.")

    raw_bytes: Optional[bytes] = None
    orig_sr: Optional[int] = None
    waveform: Optional[Any] = None

    # 1. Handle dictionary wrapper
    if isinstance(audio_input, dict):
        if "audio" in audio_input:
            inner_audio = audio_input["audio"]
            inner_sr = audio_input.get("sampling_rate")
            if isinstance(inner_audio, (np.ndarray, list, tuple)):
                waveform = np.asarray(inner_audio, dtype=np.float32)
                orig_sr = int(inner_sr) if inner_sr else target_sampling_rate
            else:
                return decode_and_resample_audio(inner_audio, target_sampling_rate=target_sampling_rate)
        elif "bytes" in audio_input:
            raw_bytes = bytes(audio_input["bytes"])
        elif "path" in audio_input:
            audio_input = audio_input["path"]
        else:
            raise InputProcessingError(
                f"Audio dictionary must contain 'audio', 'bytes', or 'path' key, got keys: {list(audio_input.keys())}"
            )

    # 2. Handle already decoded NumPy array or list
    if waveform is None and isinstance(audio_input, (np.ndarray, list, tuple)):
        waveform = np.asarray(audio_input, dtype=np.float32)
        if waveform.size == 0:
            raise InputProcessingError("Audio waveform array is empty (0 samples).")
        orig_sr = target_sampling_rate

    # 3. Handle file path
    if waveform is None and raw_bytes is None and isinstance(audio_input, (str, os.PathLike)):
        path_str = str(audio_input).strip()
        if not path_str:
            raise InputProcessingError("Audio file path cannot be empty.")
        if not os.path.isfile(path_str):
            raise InputProcessingError(f"Audio file does not exist: '{path_str}'")
        try:
            with open(path_str, "rb") as f:
                raw_bytes = f.read()
            if not raw_bytes:
                raise InputProcessingError(f"Audio file '{path_str}' is empty (0 bytes).")
        except OSError as e:
            raise InputProcessingError(f"Failed to read audio file '{path_str}': {e}") from e

    # 4. Handle raw bytes / bytearray
    if waveform is None and raw_bytes is None:
        if isinstance(audio_input, (bytes, bytearray, memoryview)):
            raw_bytes = bytes(audio_input)
            if not raw_bytes:
                raise InputProcessingError("Audio binary payload is empty (0 bytes).")
        else:
            raise InputProcessingError(
                f"Unsupported audio input type: {type(audio_input).__name__}. "
                f"Expected bytes, audio file path, or NumPy waveform array."
            )

    # 5. Validate binary audio payload and decode
    if raw_bytes is not None:
        valid_hdr, fmt = is_valid_audio_payload(raw_bytes)
        if not valid_hdr:
            raise InputProcessingError(
                f"Unsupported or malformed audio data. Payload does not contain a recognized "
                f"audio header (expected WAV, MP3, FLAC, OGG). First 8 bytes: {raw_bytes[:8]!r}"
            )

        try:
            import soundfile as sf
        except ImportError as e:
            raise DependencyMissingError(
                f"soundfile is required for audio decoding: {e}. Install via: pip install soundfile",
                missing_packages=["soundfile"],
                details={"error": str(e)}
            ) from e

        try:
            data, sr = sf.read(io.BytesIO(raw_bytes))
            orig_sr = sr
            waveform = np.asarray(data, dtype=np.float32)
        except Exception as sf_err:
            # Fallback to librosa for decoding if soundfile failed (e.g. non-standard MP3 streams)
            try:
                import librosa
                data, sr = librosa.load(io.BytesIO(raw_bytes), sr=None, mono=False)
                orig_sr = sr
                waveform = np.asarray(data, dtype=np.float32)
            except Exception as lib_err:
                raise InputProcessingError(
                    f"Failed to decode {fmt.upper()} audio data: {sf_err}. Fallback librosa error: {lib_err}",
                    details={"format": fmt, "primary_error": str(sf_err), "fallback_error": str(lib_err)}
                ) from sf_err

    if waveform is None or waveform.size == 0:
        raise InputProcessingError("Decoded audio waveform contains 0 samples.")

    # 6. Convert multi-channel / stereo to mono
    if waveform.ndim > 1:
        if waveform.shape[0] == 2 and waveform.shape[1] > 2:
            # shape (2, N)
            waveform = np.mean(waveform, axis=0)
        else:
            # shape (N, 2)
            waveform = np.mean(waveform, axis=-1)

    waveform = waveform.squeeze()
    if waveform.ndim != 1:
        waveform = waveform.flatten()

    waveform = np.asarray(waveform, dtype=np.float32)

    # 7. Resample to target sampling rate if needed
    effective_sr = orig_sr or target_sampling_rate
    if effective_sr != target_sampling_rate:
        try:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=effective_sr, target_sr=target_sampling_rate)
        except Exception:
            try:
                import scipy.signal
                num_target_samples = int(round(len(waveform) * float(target_sampling_rate) / effective_sr))
                waveform = scipy.signal.resample(waveform, num_target_samples)
            except Exception as resample_err:
                raise InputProcessingError(
                    f"Failed to resample audio from {effective_sr}Hz to {target_sampling_rate}Hz: {resample_err}"
                ) from resample_err

        waveform = np.asarray(waveform, dtype=np.float32)

    duration_sec = float(len(waveform)) / float(target_sampling_rate)
    return waveform, duration_sec
