import os
import threading
from functools import lru_cache
from typing import Optional
from collections.abc import Mapping

import numpy as np


_model_lock = threading.Lock()
_model = None


def get_project_root() -> str:
    return os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))


def get_model_path() -> str:
    # Strictly require a checkpoint .pt file
    root = get_project_root()
    for fname in ("checkpoint_epoch_30.pt", "checkpoint_epocj_30.pt"):
        candidate = os.path.join(root, fname)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Required checkpoint .pt not found. Expected 'checkpoint_epoch_30.pt' (or 'checkpoint_epocj_30.pt') at project root."
    )


class TorchModelWrapper:
    """Thin wrapper to expose a Keras-like predict(x) for a PyTorch model."""

    def __init__(self, torch_module):
        self._torch_module = torch_module

    def predict(self, x: np.ndarray) -> np.ndarray:
        # Lazy import to avoid torch dependency unless needed
        try:
            import torch  # type: ignore
        except ImportError as e:
            raise ImportError(
                "PyTorch is required to use a .pt checkpoint. Please install 'torch'."
            ) from e

        with torch.no_grad():
            tensor = torch.from_numpy(x.astype(np.float32))
            # Convert from NHWC (e.g., 1, 40, 5, 1) to NCHW if needed
            if tensor.ndim == 4 and tensor.shape[-1] == 1:
                tensor = tensor.permute(0, 3, 1, 2)
            out = self._torch_module(tensor)
            # Handle dict-like outputs (e.g., {"logits": ..., ...})
            if isinstance(out, Mapping):
                for key in ("logits", "pred", "output", "outputs"):
                    if key in out:
                        out = out[key]
                        break
                else:
                    # Fallback to first value
                    try:
                        out = next(iter(out.values()))
                    except Exception as exc:  # noqa: BLE001
                        raise RuntimeError("Torch model returned a mapping we cannot interpret.") from exc
            if isinstance(out, (tuple, list)) and len(out) > 0:
                out = out[0]
            if hasattr(out, "detach"):
                out = out.detach().cpu().numpy()
            out = np.array(out)
            # Convert logits to probabilities if possible
            if out.ndim == 2:
                out = out - out.max(axis=1, keepdims=True)
                np.exp(out, out)
                denom = out.sum(axis=1, keepdims=True) + 1e-9
                out = out / denom
            return out


def load_shared_model():
    global _model
    with _model_lock:
        if _model is None:
            model_path = get_model_path()
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")

            # Choose loader based on extension
            if model_path.lower().endswith(".pt"):
                # Require torch for .pt loading
                try:
                    import torch  # type: ignore
                except ImportError as e:
                    raise ImportError(
                        "PyTorch is required to load .pt models. Please install 'torch'."
                    ) from e

                # Try TorchScript first, then standard torch.load
                module = None
                try:
                    module = torch.jit.load(model_path, map_location="cpu")
                except Exception as ex_ts:
                    try:
                        module = torch.load(model_path, map_location="cpu")
                    except Exception as ex_pickle:
                        raise RuntimeError(
                            f"Failed to load checkpoint '{model_path}' as TorchScript or pickled module"
                        ) from ex_pickle

                # If a checkpoint mapping/list was loaded, try to extract a callable model recursively
                def _extract_model(obj):
                    # If it's already a callable module (e.g., torch.nn.Module or TorchScript), use it
                    if callable(obj) and not isinstance(obj, Mapping):
                        return obj
                    # Named common keys inside mappings
                    if isinstance(obj, Mapping):
                        for key in ("model", "module", "net", "student", "ema", "model_ema", "generator", "discriminator"):
                            if key in obj:
                                found = _extract_model(obj[key])
                                if found is not None:
                                    return found
                        # Search all values
                        for value in obj.values():
                            found = _extract_model(value)
                            if found is not None:
                                return found
                        return None
                    # Sequences
                    if isinstance(obj, (list, tuple)):
                        for item in obj:
                            found = _extract_model(item)
                            if found is not None:
                                return found
                        return None
                    return None

                extracted = _extract_model(module)
                if extracted is None:
                    # Helper: find a nested state_dict anywhere in the checkpoint
                    def _find_state_dict(obj):
                        try:
                            import torch  # type: ignore
                            from collections import OrderedDict  # type: ignore
                        except Exception:
                            return None

                        # Direct mapping that looks like a state_dict
                        if isinstance(obj, Mapping):
                            # Common direct keys
                            for key in ("state_dict", "model_state_dict", "params", "weights"):
                                if key in obj and isinstance(obj[key], Mapping):
                                    return obj[key]
                            # Heuristic: many keys with dots and tensor-like values
                            values = list(obj.values())
                            if values and all(isinstance(k, str) for k in obj.keys()):
                                tensor_like = 0
                                for v in values[:10]:
                                    if hasattr(v, "shape") or str(type(v)).endswith("Tensor"):
                                        tensor_like += 1
                                if tensor_like >= max(1, len(values) // 2):
                                    return obj
                            # Recurse into values
                            for v in obj.values():
                                found = _find_state_dict(v)
                                if found is not None:
                                    return found
                            return None
                        # Objects exposing .state_dict()
                        if hasattr(obj, "state_dict") and callable(getattr(obj, "state_dict")):
                            try:
                                sd = obj.state_dict()
                                if isinstance(sd, Mapping):
                                    return sd
                            except Exception:
                                pass
                        # Sequences
                        if isinstance(obj, (list, tuple)):
                            for item in obj:
                                found = _find_state_dict(item)
                                if found is not None:
                                    return found
                        return None

                    # If checkpoint contains a state_dict (directly or nested), build a minimal CNN and load it
                    state = _find_state_dict(module)
                    if state is not None:
                        # Infer number of classes if available at top-level mapping
                        def _infer_num_classes(mapping):
                            if isinstance(mapping, Mapping):
                                for key in ("num_classes", "n_classes", "classes", "num_outputs", "output_dim", "num_labels"):
                                    if key in mapping:
                                        try:
                                            return int(mapping[key])
                                        except Exception:
                                            continue
                            return 10

                        num_classes = _infer_num_classes(module)

                        import torch.nn as nn  # type: ignore

                        class TorchFallbackCNN(nn.Module):
                            def __init__(self, num_classes: int):
                                super().__init__()
                                self.features = nn.Sequential(
                                    nn.Conv2d(1, 16, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(16, 32, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(inplace=True),
                                )
                                self.pool = nn.AdaptiveAvgPool2d((5, 1))
                                self.classifier = nn.Linear(32 * 5 * 1, num_classes)

                            def forward(self, x):  # x: (N, C=1, H=40, W=5)
                                x = self.features(x)
                                x = self.pool(x)
                                x = x.view(x.size(0), -1)
                                x = self.classifier(x)
                                return x

                        constructed = TorchFallbackCNN(num_classes=num_classes)
                        try:
                            constructed.load_state_dict(state, strict=False)
                        except Exception:
                            pass
                        constructed.eval()
                        extracted = constructed
                    else:
                        raise RuntimeError(
                            "Unable to extract a callable model from the .pt checkpoint. Provide a TorchScript file or a pickled nn.Module."
                        )

                module = extracted

                # Final sanity: ensure we have a callable torch module
                if not callable(module):
                    raise RuntimeError(
                        f"Loaded object from {model_path} is not callable (type={type(module)}). Provide a TorchScript or full model object."
                    )

                # Put into eval mode if available
                if hasattr(module, "eval"):
                    module.eval()
                _model = TorchModelWrapper(module)
            else:
                raise RuntimeError("Only .pt checkpoints are supported by this build.")
        return _model


@lru_cache(maxsize=1)
def get_noise_dir() -> str:
    root = get_project_root()
    return os.path.join(root, "noise")





