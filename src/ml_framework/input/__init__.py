"""
Input processing subsystem for the ML Framework.
Provides multi-source readers (local file, object storage, Kafka, direct)
and orchestrating processor under the BaseInputProcessor contract.
"""

from .readers import (
    BaseInputReader,
    LocalFileReader,
    ObjectStorageReader,
    KafkaTopicReader,
    DirectInputReader,
)
from .processor import InputProcessor
from .audio import decode_and_resample_audio, is_valid_audio_payload

__all__ = [
    "BaseInputReader",
    "LocalFileReader",
    "ObjectStorageReader",
    "KafkaTopicReader",
    "DirectInputReader",
    "InputProcessor",
    "decode_and_resample_audio",
    "is_valid_audio_payload",
]

