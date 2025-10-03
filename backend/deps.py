import os
import threading
from functools import lru_cache
from typing import Optional

import numpy as np
from tensorflow.keras.models import load_model


_model_lock = threading.Lock()
_model = None


def get_project_root() -> str:
    return os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))


def get_model_path() -> str:
    # Prefer env, fallback to root model.h5
    model_env = os.getenv("MODEL_PATH")
    if model_env and os.path.exists(model_env):
        return model_env
    root = get_project_root()
    candidate = os.path.join(root, "model.h5")
    return candidate


def load_shared_model():
    global _model
    with _model_lock:
        if _model is None:
            model_path = get_model_path()
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
            _model = load_model(model_path)
        return _model


@lru_cache(maxsize=1)
def get_noise_dir() -> str:
    root = get_project_root()
    return os.path.join(root, "noise")





