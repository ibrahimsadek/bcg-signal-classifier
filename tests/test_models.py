# -*- coding: utf-8 -*-
"""Tests for models module."""

import numpy as np
import pytest
import tensorflow as tf

from bcg_signal_classifier.models import build_cnn, build_transformer, build_model


def test_build_cnn_returns_model():
    """Test that build_cnn returns a compiled Keras model."""
    model = build_cnn(input_len=50, lr=1e-4, dropout=0.2)
    assert isinstance(model, tf.keras.Model)
    assert model.name == "cnn_logits"


def test_build_cnn_correct_shapes():
    """Test that CNN model has correct input/output shapes."""
    model = build_cnn(input_len=50, lr=1e-4, dropout=0.2)
    
    # Test forward pass
    X = np.random.randn(10, 50, 1).astype(np.float32)
    logits = model.predict(X, verbose=0)
    
    assert logits.shape == (10, 2), "Output should be (batch_size, 2) logits"


def test_build_transformer_returns_model():
    """Test that build_transformer returns a compiled Keras model."""
    model = build_transformer(input_len=50, lr=1e-4, dropout=0.2)
    assert isinstance(model, tf.keras.Model)
    assert model.name == "conv_transformer_logits"


def test_build_transformer_correct_shapes():
    """Test that Transformer model has correct input/output shapes."""
    model = build_transformer(input_len=50, lr=1e-4, dropout=0.2)
    
    # Test forward pass
    X = np.random.randn(10, 50, 1).astype(np.float32)
    logits = model.predict(X, verbose=0)
    
    assert logits.shape == (10, 2), "Output should be (batch_size, 2) logits"


def test_build_model_dispatches_cnn():
    """Test that build_model correctly dispatches to CNN."""
    model = build_model("cnn", input_len=50, lr=1e-4, dropout=0.2)
    assert model.name == "cnn_logits"


def test_build_model_dispatches_transformer():
    """Test that build_model correctly dispatches to Transformer."""
    model = build_model("transformer", input_len=50, lr=1e-4, dropout=0.2)
    assert model.name == "conv_transformer_logits"


def test_build_model_raises_on_unknown():
    """Test that build_model raises ValueError for unknown model type."""
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("unknown_model", input_len=50, lr=1e-4, dropout=0.2)
