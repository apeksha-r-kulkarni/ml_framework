"""
Triton Python Backend Model for NVIDIA NeMo Conformer-CTC ASR serving.
Handles audio signal tensor input and generates transcript text.
"""

import gc
import json
import logging
import os
import tempfile

import numpy as np
import soundfile as sf
import torch
import nemo.collections.asr as nemo_asr
import triton_python_backend_utils as pb_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nemo_conformer_python")


class TritonPythonModel:
    """
    Triton Python model implementing KServe v2 inference for NeMo Conformer-CTC.
    """

    def initialize(self, args):
        """
        Loads the NeMo model from the mounted checkpoint directory or model repository.
        """
        self.model_config = json.loads(args["model_config"])
        self.model_name = args.get("model_name", "nemo_conformer_python")
        self.target_sample_rate = 16000

        # Candidate paths for model checkpoint
        candidate_paths = [
            os.environ.get("NEMO_MODEL_PATH", ""),
            "/model/stt_en_conformer_ctc_large.nemo",
            "/models/nemo_conformer_python/1/stt_en_conformer_ctc_large.nemo",
            "/workspace/stt_en_conformer_ctc_large.nemo",
        ]

        # Check /model directory for any .nemo file
        if os.path.isdir("/model"):
            for f in os.listdir("/model"):
                if f.endswith(".nemo"):
                    candidate_paths.append(os.path.join("/model", f))

        model_path = None
        for path in candidate_paths:
            if path and os.path.isfile(path):
                model_path = path
                break

        if not model_path:
            raise RuntimeError(
                f"[{self.model_name}] Model checkpoint not found in candidate paths: {candidate_paths}"
            )

        logger.info(f"[{self.model_name}] Loading NeMo model from '{model_path}' on CPU...")
        try:
            self.model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
                restore_path=model_path,
                map_location="cpu",
            )
            self.model.eval()
            logger.info(
                f"[{self.model_name}] NeMo model successfully loaded. Target sample rate: {self.target_sample_rate}"
            )
        except Exception as e:
            raise RuntimeError(
                f"[{self.model_name}] Failed to restore NeMo model from '{model_path}': {e}"
            ) from e

    def execute(self, requests):
        """
        Executes speech transcription inference on incoming audio batches.
        """
        responses = []

        for request in requests:
            # 1. Extract AUDIO_SIGNAL tensor
            audio_tensor = pb_utils.get_input_tensor_by_name(request, "AUDIO_SIGNAL")
            if audio_tensor is None:
                error = getattr(pb_utils, "TritonError", RuntimeError)("Missing required input tensor 'AUDIO_SIGNAL'")
                resp_cls = getattr(pb_utils, "InferenceResponse")
                responses.append(resp_cls(output_tensors=[], error=error))
                continue

            audio_signal = audio_tensor.as_numpy().flatten().astype(np.float32)

            # 2. Extract SAMPLE_RATE tensor
            sr_tensor = pb_utils.get_input_tensor_by_name(request, "SAMPLE_RATE")
            if sr_tensor is not None:
                sample_rate = int(sr_tensor.as_numpy().flatten()[0])
            else:
                sample_rate = self.target_sample_rate

            # 3. Transcribe via temporary WAV file
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_wav:
                    sf.write(tmp_wav.name, audio_signal, sample_rate, subtype="PCM_16")
                    hypotheses = self.model.transcribe([tmp_wav.name], verbose=False)
                    transcript = hypotheses[0] if (hypotheses and len(hypotheses) > 0) else ""

                # 4. Create TRANSCRIPT output tensor
                out_tensor = pb_utils.Tensor(
                    "TRANSCRIPT",
                    np.array([transcript], dtype=np.object_),
                )
                resp_cls = getattr(pb_utils, "InferenceResponse")
                responses.append(resp_cls(output_tensors=[out_tensor]))

            except Exception as exc:
                logger.error(f"[{self.model_name}] Inference execution failed: {exc}", exc_info=True)
                error = getattr(pb_utils, "TritonError", RuntimeError)(f"ASR inference failure: {exc}")
                resp_cls = getattr(pb_utils, "InferenceResponse")
                responses.append(resp_cls(output_tensors=[], error=error))

        return responses

    def finalize(self):
        """
        Releases model weights and frees system memory on model unload or container shutdown.
        """
        logger.info(f"[{self.model_name}] Finalizing model and cleaning memory.")
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
