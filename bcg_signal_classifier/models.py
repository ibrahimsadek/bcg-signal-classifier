# -*- coding: utf-8 -*-
"""Model building functions for BCG signal classification."""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn(input_len: int, lr: float, dropout: float) -> tf.keras.Model:
    """Build a CNN model for BCG classification.

    Args:
        input_len: Length of input signals.
        lr: Learning rate.
        dropout: Dropout rate.

    Returns:
        Compiled Keras model outputting logits.
    """
    inp = layers.Input(shape=(input_len, 1))
    x = layers.Conv1D(64, 7, padding="same", activation="relu")(inp)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    logits = layers.Dense(2, activation=None, dtype="float32")(x)

    m = models.Model(inp, logits, name="cnn_logits")
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return m


def transformer_block(x: tf.Tensor, num_heads: int, key_dim: int, ff_dim: int, dropout: float) -> tf.Tensor:
    """Transformer encoder block.

    Args:
        x: Input tensor.
        num_heads: Number of attention heads.
        key_dim: Dimension of attention keys.
        ff_dim: Feedforward dimension.
        dropout: Dropout rate.

    Returns:
        Output tensor after transformer block.
    """
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dropout(dropout)(ff)
    ff = layers.Dense(x.shape[-1])(ff)
    x = layers.Add()([x, ff])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x


def build_transformer(input_len: int, lr: float, dropout: float) -> tf.keras.Model:
    """Build a Transformer model for BCG classification.

    Args:
        input_len: Length of input signals.
        lr: Learning rate.
        dropout: Dropout rate.

    Returns:
        Compiled Keras model outputting logits.
    """
    inp = layers.Input(shape=(input_len, 1))

    x = layers.Conv1D(64, 5, padding="same", activation="relu")(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)

    pos = tf.range(start=0, limit=input_len, delta=1)
    pos_emb = layers.Embedding(input_dim=input_len, output_dim=64)(pos)
    x = x + pos_emb

    x = transformer_block(x, num_heads=4, key_dim=16, ff_dim=128, dropout=dropout)
    x = transformer_block(x, num_heads=4, key_dim=16, ff_dim=128, dropout=dropout)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    logits = layers.Dense(2, activation=None, dtype="float32")(x)

    m = models.Model(inp, logits, name="conv_transformer_logits")
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return m


def build_model(model_name: str, input_len: int, lr: float, dropout: float) -> tf.keras.Model:
    """Build a model based on the model name.

    Args:
        model_name: Model type ('cnn' or 'transformer').
        input_len: Length of input signals.
        lr: Learning rate.
        dropout: Dropout rate.

    Returns:
        Compiled Keras model outputting logits.

    Raises:
        ValueError: If model_name is unknown.
    """
    if model_name == "cnn":
        return build_cnn(input_len, lr=lr, dropout=dropout)
    if model_name == "transformer":
        return build_transformer(input_len, lr=lr, dropout=dropout)
    raise ValueError(f"Unknown model: {model_name}")
