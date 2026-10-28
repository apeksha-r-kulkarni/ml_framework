"""
ML public API facade.

Enables the required meeting interface:
    from ml_framework import ml

    model = ml.load(model_source="...", config=...)
    result = model.process(input_data)
    model.unload()
    model.stop()
"""

from typing import Any, Dict, Optional, Union

from .config.model_config import FrameworkConfig, InputConfig, StorageConfig
from .execution_manager import ExecutionManager
from .interfaces.base import (
    BaseEnvironmentManager,
    BaseInputProcessor,
    BaseModelLoader,
    BaseInferenceEngine,
    BaseOutputProcessor,
    BaseStatusTracker,
    BaseResourceManager,
)
from .input.processor import InputProcessor
from .environment.environment_manager import EnvironmentManager
from .models.model_loader import ModelLoader
from .errors.exceptions import ConfigurationError


def load(
    model_source: Optional[str] = None,
    config: Optional[Union[FrameworkConfig, Dict[str, Any]]] = None,
    input_source: Optional[Any] = None,
    env_manager: Optional[BaseEnvironmentManager] = None,
    input_processor: Optional[BaseInputProcessor] = None,
    model_loader: Optional[BaseModelLoader] = None,
    inference_engine: Optional[BaseInferenceEngine] = None,
    output_processor: Optional[BaseOutputProcessor] = None,
    status_tracker: Optional[BaseStatusTracker] = None,
    resource_manager: Optional[BaseResourceManager] = None,
    model_bucket: Optional[str] = None,
    model_key: Optional[str] = None,
    model_dir: Optional[str] = None,
    model_id: Optional[str] = None,
    task: Optional[str] = None,
    input_type: Optional[str] = None,
    **kwargs
) -> ExecutionManager:
    """
    Loads and initializes an ML model pipeline according to the public API specification:
        from ml_framework import ml
        model = ml.load(...)
        result = model.process(...)
        model.unload()
        model.stop()

    1. Parses and validates FrameworkConfig.
    2. Validates input source configuration (if provided) before starting model loading.
    3. Validates environment dependencies and hardware compatibility offline.
    4. Instantiates and initializes the ExecutionManager.
    5. Returns the ExecutionManager instance as the active model controller.
    """
    if model_bucket:
        kwargs["model_bucket"] = model_bucket
    if model_key:
        kwargs["model_key"] = model_key
    if model_dir:
        kwargs["model_dir"] = model_dir
    if model_id:
        kwargs["model_id"] = model_id
    if task:
        kwargs["task"] = task
    if input_type:
        kwargs["input_type"] = input_type

    # 1. Parse configuration
    if config is None:
        cfg = FrameworkConfig.from_dict({"model_source": model_source, **kwargs})
    elif isinstance(config, dict):
        cfg_dict = dict(config)
        if model_source and "model_source" not in cfg_dict:
            cfg_dict["model_source"] = model_source
        cfg_dict.update(kwargs)
        cfg = FrameworkConfig.from_dict(cfg_dict)
    elif isinstance(config, FrameworkConfig):
        cfg = config
        if model_source:
            cfg.model_source = model_source
        if model_bucket:
            cfg.model_bucket = model_bucket
        if model_key:
            cfg.model_key = model_key
        if model_dir:
            cfg.model_dir = model_dir
            if not cfg.model_key:
                cfg.model_key = model_dir
        if model_id:
            cfg.model_id = model_id
            if not cfg.model_name:
                cfg.model_name = model_id
        if task:
            cfg.task = task
        if input_type:
            cfg.input_type = input_type
        cfg.validate()
    else:
        raise ConfigurationError(
            f"Expected FrameworkConfig or dict, got {type(config).__name__}"
        )

    if input_source is not None:
        cfg.input_config.source = input_source
        cfg.input_source = input_source

    # 2. Instantiate default components if not explicitly injected
    resolved_env_mgr = env_manager if env_manager is not None else EnvironmentManager()
    resolved_input_proc = input_processor if input_processor is not None else InputProcessor(
        config=cfg.input_config,
        storage_config=cfg.storage_config,
    )
    if model_loader is not None:
        resolved_model_loader = model_loader
    elif (cfg.model_source and cfg.model_source != "local://models/dummy") or (cfg.model_bucket and cfg.model_key):
        resolved_model_loader = ModelLoader(
            env_manager=resolved_env_mgr,
            storage_config=cfg.storage_config,
        )
    else:
        resolved_model_loader = None

    # 3. Resolve inference engine (defaults to TritonInferenceEngine as exclusive engine)
    if inference_engine is not None:
        resolved_inference_engine = inference_engine
    else:
        from .inference.triton_engine import TritonInferenceEngine
        if not cfg.triton_config.model_name and cfg.model_name:
            cfg.triton_config.model_name = cfg.model_name
        resolved_inference_engine = TritonInferenceEngine(config=cfg.triton_config)

    # 4. Instantiate ExecutionManager
    manager = ExecutionManager(
        config=cfg,
        env_manager=resolved_env_mgr,
        input_processor=resolved_input_proc,
        model_loader=resolved_model_loader,
        inference_engine=resolved_inference_engine,
        output_processor=output_processor,
        status_tracker=status_tracker,
        resource_manager=resource_manager,
    )

    # 4. Initialize manager (triggers pre-load input validation, env check, and model load)
    manager.initialize()

    return manager


def validate_input(
    input_source: Any,
    config: Optional[InputConfig] = None,
    storage_config: Optional[StorageConfig] = None
) -> bool:
    """
    Validates input source accessibility and format before execution.
    Raises InvalidInputSourceError or InputProcessingError if invalid.
    """
    proc = InputProcessor(config=config, storage_config=storage_config)
    return proc.validate_source(input_source)


def validate_environment(
    config: Optional[FrameworkConfig] = None,
    model_metadata: Optional[Any] = None
) -> bool:
    """
    Validates runtime dependencies and device compatibility offline.
    Raises DependencyMissingError or HardwareIncompatibleError if invalid.
    """
    env_mgr = EnvironmentManager()
    cfg = config or FrameworkConfig()
    return env_mgr.validate_environment(cfg, model_metadata=model_metadata)
