# -*- coding: utf-8 -*-
"""Save and load trained models, calibrators, and inference configuration.

This module makes trained pipelines reusable: after training, a deployable
model is written to disk together with the post-hoc calibrator and the exact
preprocessing settings required to reproduce the training-time feature
extraction. The same artifacts are reloaded at inference time.

Artifact layout (inside ``model_dir``):
    model.keras            Keras model (logits output).
    calibrator.json        Calibrator type + parameters (or ``{"type": "none"}``).
    inference_config.json  Preprocessing + model metadata and label mapping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import tensorflow as tf

from bcg_signal_classifier.calibration import PlattScalerBinary, TemperatureScaler
from bcg_signal_classifier.config import Config

Calibrator = Union[TemperatureScaler, PlattScalerBinary]

MODEL_FILENAME = "model.keras"
CALIBRATOR_FILENAME = "calibrator.json"
INFERENCE_CONFIG_FILENAME = "inference_config.json"

# Stable mapping between class indices and human-readable labels.
LABEL_MAP = {0: "NonBCG", 1: "BCG"}


def calibrator_to_dict(calibrator: Optional[Calibrator]) -> dict:
    """Serialize a fitted calibrator to a JSON-compatible dictionary.

    Args:
        calibrator: Fitted ``TemperatureScaler``, ``PlattScalerBinary``, or None.

    Returns:
        Dictionary describing the calibrator type and its parameters.

    Raises:
        TypeError: If the calibrator type is not recognized.
    """
    if calibrator is None:
        return {"type": "none"}
    if isinstance(calibrator, TemperatureScaler):
        return {"type": "temperature", "temperature": float(calibrator.temperature_)}
    if isinstance(calibrator, PlattScalerBinary):
        return {
            "type": "platt",
            "coef": np.asarray(calibrator.clf.coef_, dtype=float).tolist(),
            "intercept": np.asarray(calibrator.clf.intercept_, dtype=float).tolist(),
            "classes": np.asarray(calibrator.clf.classes_, dtype=int).tolist(),
        }
    raise TypeError(f"Unknown calibrator type: {type(calibrator)!r}")


def calibrator_from_dict(d: dict) -> Optional[Calibrator]:
    """Reconstruct a calibrator from its serialized dictionary.

    Args:
        d: Dictionary produced by :func:`calibrator_to_dict`.

    Returns:
        A fitted calibrator instance, or None if the type is ``"none"``.

    Raises:
        ValueError: If the calibrator type is unknown.
    """
    ctype = d.get("type", "none")
    if ctype == "none":
        return None
    if ctype == "temperature":
        scaler = TemperatureScaler()
        scaler.temperature_ = float(d["temperature"])
        return scaler
    if ctype == "platt":
        scaler = PlattScalerBinary()
        # Restore the underlying logistic regression without refitting.
        scaler.clf.coef_ = np.asarray(d["coef"], dtype=float)
        scaler.clf.intercept_ = np.asarray(d["intercept"], dtype=float)
        scaler.clf.classes_ = np.asarray(d["classes"], dtype=int)
        scaler.clf.n_features_in_ = scaler.clf.coef_.shape[1]
        return scaler
    raise ValueError(f"Unknown calibrator type: {ctype}")


def inference_config_from_cfg(cfg: Config, model_name: str) -> dict:
    """Extract the subset of the config needed to reproduce preprocessing.

    Args:
        cfg: Configuration used during training.
        model_name: Model architecture name ('cnn' or 'transformer').

    Returns:
        JSON-compatible dictionary with preprocessing and model metadata.
    """
    return {
        "model_name": model_name,
        "fs": float(cfg.fs),
        "hp_order": int(cfg.hp_order),
        "hp_ripple": float(cfg.hp_ripple),
        "hp_cutoff_hz": float(cfg.hp_cutoff_hz),
        "lp_order": int(cfg.lp_order),
        "lp_ripple": float(cfg.lp_ripple),
        "lp_cutoff_hz": float(cfg.lp_cutoff_hz),
        "target_len": int(cfg.target_len),
        "calibration": str(cfg.calibration),
        "label_map": {str(k): v for k, v in LABEL_MAP.items()},
    }


def save_artifacts(
    model_dir: Path,
    model: tf.keras.Model,
    calibrator: Optional[Calibrator],
    cfg: Config,
    model_name: str,
) -> Path:
    """Persist a deployable model, its calibrator, and inference metadata.

    Args:
        model_dir: Destination directory (created if missing).
        model: Trained Keras model outputting logits.
        calibrator: Fitted calibrator or None.
        cfg: Configuration used during training.
        model_name: Model architecture name ('cnn' or 'transformer').

    Returns:
        The ``model_dir`` path that artifacts were written to.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model.save(model_dir / MODEL_FILENAME)

    with open(model_dir / CALIBRATOR_FILENAME, "w", encoding="utf-8") as f:
        json.dump(calibrator_to_dict(calibrator), f, indent=2)

    with open(model_dir / INFERENCE_CONFIG_FILENAME, "w", encoding="utf-8") as f:
        json.dump(inference_config_from_cfg(cfg, model_name), f, indent=2)

    return model_dir


def load_artifacts(model_dir: Path):
    """Load a trained model, calibrator, and inference configuration.

    Args:
        model_dir: Directory previously written by :func:`save_artifacts`.

    Returns:
        Tuple ``(model, calibrator, inference_config)`` where ``model`` is a
        Keras model, ``calibrator`` is a fitted calibrator or None, and
        ``inference_config`` is the metadata dictionary.

    Raises:
        FileNotFoundError: If any required artifact is missing.
    """
    model_dir = Path(model_dir)
    model_path = model_dir / MODEL_FILENAME
    calib_path = model_dir / CALIBRATOR_FILENAME
    cfg_path = model_dir / INFERENCE_CONFIG_FILENAME

    for required in (model_path, calib_path, cfg_path):
        if not required.exists():
            raise FileNotFoundError(f"Missing model artifact: {required}")

    model = tf.keras.models.load_model(model_path, compile=False)

    with open(calib_path, "r", encoding="utf-8") as f:
        calibrator = calibrator_from_dict(json.load(f))

    with open(cfg_path, "r", encoding="utf-8") as f:
        inference_config = json.load(f)

    return model, calibrator, inference_config
