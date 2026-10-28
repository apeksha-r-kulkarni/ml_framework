"""
spaCy Model Framework Adapter.

Encapsulates offline loading, device placement, and lifecycle management
for spaCy NLP pipelines (e.g. en_core_web_sm, custom pipelines).
"""

import os
import gc
import logging
from typing import Any, Optional

from ..interfaces.base import BaseModelAdapter
from ..config.model_config import FrameworkConfig
from ..errors.exceptions import (
    ModelLoadError,
    DependencyMissingError,
    HardwareIncompatibleError,
)

logger = logging.getLogger("ml_framework.models.spacy")


class SpacyModelAdapter(BaseModelAdapter):
    """
    Framework adapter for spaCy models and pipelines.
    Enforces strictly local, offline loading (spacy.load(local_model_dir)).
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "cpu",
        architecture: str = "spaCyPipeline",
        model_type: str = "spacy",
        config: Optional[FrameworkConfig] = None,
        nlp: Optional[Any] = None,
    ):
        self.model_dir = model_dir
        self.device = device
        self.architecture = architecture
        self.model_type = model_type
        self.config = config
        self.nlp = nlp
        self._is_loaded = nlp is not None

    @property
    def is_loaded(self) -> bool:
        """Returns True if the pipeline is loaded and active in memory."""
        return self._is_loaded and self.nlp is not None

    def load(self) -> "SpacyModelAdapter":
        """
        Loads spaCy pipeline from local directory only.
        Guarantees strictly offline execution without downloading packages or models.
        """
        if self._is_loaded and self.nlp is not None:
            return self

        try:
            import spacy
        except ImportError as e:
            raise DependencyMissingError(
                f"Required dependency 'spacy' is missing: {e}. "
                f"Install via: pip install '.[spacy]' or pip install spacy>=3.5.0",
                missing_packages=["spacy"],
                details={"error": str(e), "missing_packages": ["spacy"]}
            ) from e

        # Validate hardware accelerator placement
        norm_dev = (self.device or "cpu").lower().strip()
        if norm_dev in ("cuda", "gpu") or norm_dev.startswith("cuda:"):
            try:
                gpu_ok = spacy.prefer_gpu()
                if not gpu_ok:
                    raise HardwareIncompatibleError(
                        f"GPU device '{self.device}' requested for spaCy, but GPU is not available or cupy is not installed.",
                        details={"device": self.device}
                    )
            except Exception as hw_err:
                if isinstance(hw_err, HardwareIncompatibleError):
                    raise
                raise HardwareIncompatibleError(
                    f"Failed to initialize GPU for spaCy (device='{self.device}'): {hw_err}",
                    details={"device": self.device, "error": str(hw_err)}
                ) from hw_err

        # Load pipeline offline from local directory
        try:
            nlp = spacy.load(self.model_dir)
        except Exception as load_err:
            raise ModelLoadError(
                f"Failed to load spaCy pipeline from '{self.model_dir}': {load_err}. "
                f"Ensure meta.json or config.cfg is present and valid.",
                details={"model_dir": self.model_dir, "error": str(load_err)}
            ) from load_err

        self.nlp = nlp
        self._is_loaded = True
        logger.info(
            f"spaCy pipeline loaded successfully from '{self.model_dir}' "
            f"(name={getattr(nlp, 'meta', {}).get('name', self.model_type)}, device={self.device})."
        )
        return self

    def unload(self) -> None:
        """Safely unloads pipeline instance and frees execution memory."""
        self.nlp = None
        self._is_loaded = False
        gc.collect()
        logger.info(f"spaCy pipeline unloaded from memory (model_dir='{self.model_dir}').")

    def predict(self, preprocessed_input: Any) -> Any:
        """
        Direct inference delegation helper used when ExecutionManager runs without an InferenceEngine.
        """
        if not self.is_loaded:
            raise ModelLoadError("Cannot execute prediction on unloaded spaCy model.")
        if self.nlp is not None and callable(self.nlp):
            try:
                return self.nlp(preprocessed_input)
            except Exception:
                return preprocessed_input
        return preprocessed_input
