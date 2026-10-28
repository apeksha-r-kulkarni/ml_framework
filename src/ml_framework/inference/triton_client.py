"""
Triton Inference Server HTTP v2 (KServe v2) Client.

Provides a robust, zero-external-dependency HTTP client for communicating with
NVIDIA Triton Inference Server. Implements health checks, model readiness probes,
metadata inspection, and synchronous inference execution with retries and comprehensive error handling.
"""

import json
import time
import socket
import logging
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Union
from ..errors.exceptions import InferenceError

logger = logging.getLogger("ml_framework.inference.triton_client")


class TritonClient:
    """
    Standard HTTP client for NVIDIA Triton Inference Server using the KServe v2 REST API.
    """

    def __init__(
        self,
        endpoint_url: str = "http://localhost:8010",
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        retry_interval_seconds: float = 0.5,
    ):
        clean_url = str(endpoint_url).strip()
        if not clean_url.startswith(("http://", "https://")):
            clean_url = f"http://{clean_url}"
        self.endpoint_url = clean_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max(0, int(max_retries))
        self.retry_interval_seconds = max(0.0, float(retry_interval_seconds))

    def _build_url(self, path: str) -> str:
        clean_path = path.lstrip("/")
        return f"{self.endpoint_url}/{clean_path}"

    def _http_request(
        self,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        expected_status: Union[int, List[int]] = 200,
    ) -> Dict[str, Any]:
        """
        Executes an HTTP request to Triton with retry handling and structured error reporting.
        """
        url = self._build_url(path)
        effective_timeout = timeout if timeout is not None else self.timeout_seconds
        payload_bytes = json.dumps(data).encode("utf-8") if data is not None else None

        allowed_statuses = [expected_status] if isinstance(expected_status, int) else expected_status

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    url,
                    data=payload_bytes,
                    method=method.upper(),
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    }
                )
                with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
                    status_code = resp.getcode()
                    resp_bytes = resp.read()
                    resp_str = resp_bytes.decode("utf-8") if resp_bytes else ""
                    if status_code in allowed_statuses:
                        return json.loads(resp_str) if resp_str.strip() else {}
                    raise InferenceError(
                        f"Unexpected HTTP status {status_code} from Triton at {url}: {resp_str}",
                        details={"status_code": status_code, "url": url, "response": resp_str}
                    )

            except urllib.error.HTTPError as http_err:
                status_code = http_err.code
                err_body = http_err.read().decode("utf-8", errors="replace")
                # Do not retry 4xx errors (client errors: not ready, bad request, not found)
                if 400 <= status_code < 500:
                    raise InferenceError(
                        f"Triton returned HTTP {status_code} for {url}: {err_body}",
                        details={"status_code": status_code, "url": url, "error": err_body}
                    ) from http_err
                last_error = http_err

            except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as conn_err:
                last_error = conn_err

            if attempt < self.max_retries:
                time.sleep(self.retry_interval_seconds)

        raise InferenceError(
            f"Failed to communicate with Triton Inference Server at '{url}' after "
            f"{self.max_retries + 1} attempts. Root error: {str(last_error)}",
            details={"url": url, "attempts": self.max_retries + 1, "error": str(last_error)}
        ) from last_error

    def is_server_live(self, timeout: float = 5.0) -> bool:
        """Checks if the Triton Inference Server process is alive (GET /v2/health/live)."""
        try:
            url = self._build_url("v2/health/live")
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.getcode() == 200
        except Exception as e:
            logger.debug(f"Triton is_server_live check failed: {e}")
            return False

    def is_server_ready(self, timeout: float = 5.0) -> bool:
        """Checks if the Triton Inference Server is fully ready to accept inference requests (GET /v2/health/ready)."""
        try:
            url = self._build_url("v2/health/ready")
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.getcode() == 200
        except Exception as e:
            logger.debug(f"Triton is_server_ready check failed: {e}")
            return False

    def is_model_ready(
        self,
        model_name: str,
        model_version: str = "",
        timeout: float = 5.0
    ) -> bool:
        """Checks if a specific model or version is loaded and READY on Triton."""
        if not model_name:
            return False
        path = (
            f"v2/models/{model_name}/versions/{model_version}/ready"
            if model_version
            else f"v2/models/{model_name}/ready"
        )
        try:
            url = self._build_url(path)
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.getcode() == 200
        except Exception as e:
            logger.debug(f"Triton is_model_ready probe failed for model '{model_name}': {e}")
            return False

    def get_model_metadata(
        self,
        model_name: str,
        model_version: str = "",
        timeout: float = 5.0
    ) -> Dict[str, Any]:
        """Retrieves model metadata (input and output tensor schemas) from Triton (GET /v2/models/{model_name})."""
        path = (
            f"v2/models/{model_name}/versions/{model_version}"
            if model_version
            else f"v2/models/{model_name}"
        )
        return self._http_request("GET", path, timeout=timeout)

    def get_model_config(
        self,
        model_name: str,
        model_version: str = "",
        timeout: float = 5.0
    ) -> Dict[str, Any]:
        """Retrieves model configuration from Triton (GET /v2/models/{model_name}/config)."""
        path = (
            f"v2/models/{model_name}/versions/{model_version}/config"
            if model_version
            else f"v2/models/{model_name}/config"
        )
        return self._http_request("GET", path, timeout=timeout)

    def infer(
        self,
        model_name: str,
        inputs: List[Dict[str, Any]],
        outputs: Optional[List[Dict[str, str]]] = None,
        model_version: str = "",
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Executes a synchronous model inference request using Triton's KServe v2 REST endpoint.

        Args:
            model_name: Name of the Triton-served model.
            inputs: List of input tensor specifications: [{"name": ..., "shape": [...], "datatype": ..., "data": [...]}]
            outputs: Optional requested output tensors: [{"name": "OUTPUT_NAME"}]
            model_version: Optional model version string.
            timeout: Optional per-request timeout in seconds.

        Returns:
            Decoded JSON dictionary from Triton containing output tensors and metadata.
        """
        if not model_name or not str(model_name).strip():
            raise InferenceError("model_name is required for Triton inference execution.")
        if not inputs:
            raise InferenceError("inputs list cannot be empty for Triton inference execution.")

        path = (
            f"v2/models/{model_name}/versions/{model_version}/infer"
            if model_version
            else f"v2/models/{model_name}/infer"
        )
        request_body: Dict[str, Any] = {"inputs": inputs}
        if outputs:
            request_body["outputs"] = outputs

        return self._http_request("POST", path, data=request_body, timeout=timeout)
