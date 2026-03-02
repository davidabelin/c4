"""Supervised learning pipeline for c4 move-selection models."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from random import Random
from typing import Any

import numpy as np

from c4_storage.object_store import read_bytes, write_bytes

try:
    from sklearn.metrics import accuracy_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier
except Exception as exc:  # pragma: no cover
    accuracy_score = None
    MLPClassifier = None
    DecisionTreeClassifier = None
    SKLEARN_IMPORT_ERROR = str(exc)
else:
    SKLEARN_IMPORT_ERROR = None

SKLEARN_AVAILABLE = DecisionTreeClassifier is not None and MLPClassifier is not None and accuracy_score is not None


@dataclass(slots=True)
class TrainConfig:
    """Configuration for supervised training runs."""

    model_type: str = "decision_tree"
    lookback: int = 5
    test_size: float = 0.2
    learning_rate: float = 0.001
    hidden_layer_sizes: tuple[int, ...] = (64, 32)
    epochs: int = 200
    batch_size: int | str = "auto"
    random_state: int = 42


class FrequencyModel:
    """Simple top-row context frequency baseline."""

    def __init__(self) -> None:
        self.context_counts: dict[tuple[int, ...], np.ndarray] = {}
        self.global_counts = np.ones(7, dtype=float)

    def fit(self, contexts: list[tuple[int, ...]], y: np.ndarray) -> "FrequencyModel":
        for context, label in zip(contexts, y):
            if context not in self.context_counts:
                self.context_counts[context] = np.ones(7, dtype=float)
            self.context_counts[context][int(label)] += 1.0
            self.global_counts[int(label)] += 1.0
        return self

    def predict_context(self, context: tuple[int, ...]) -> int:
        counts = self.context_counts.get(context, self.global_counts)
        return int(np.argmax(counts))

    def predict_contexts(self, contexts: list[tuple[int, ...]]) -> np.ndarray:
        return np.asarray([self.predict_context(context) for context in contexts], dtype=int)


def _parse_board(raw: str | list[int]) -> list[int]:
    if isinstance(raw, list):
        return [int(v) for v in raw]
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError("board_before_json must decode to list")
    return [int(v) for v in payload]


def build_dataset(rows: list[dict], lookback: int) -> tuple[np.ndarray, np.ndarray, list[tuple[int, ...]]]:
    """Construct supervised features/labels from stored AI move rows."""

    if lookback <= 0:
        raise ValueError("lookback must be positive")

    grouped: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        key = (int(row["game_id"]), int(row["session_index"]))
        grouped.setdefault(key, []).append(row)

    for key in grouped:
        grouped[key] = sorted(grouped[key], key=lambda item: int(item["move_index"]))

    X: list[list[float]] = []
    y: list[int] = []
    contexts: list[tuple[int, ...]] = []

    for _key, seq in grouped.items():
        actions = [int(item["action"]) for item in seq]
        for idx in range(len(seq)):
            if idx < lookback:
                continue
            current = seq[idx]
            board = _parse_board(current["board_before_json"])
            if len(board) != 42:
                continue
            history = actions[idx - lookback : idx]
            features = [float(v) for v in board]
            # Append one-hot encoded action history.
            for action in history:
                one_hot = [0.0] * 7
                if 0 <= int(action) < 7:
                    one_hot[int(action)] = 1.0
                features.extend(one_hot)
            X.append(features)
            y.append(int(current["action"]))
            contexts.append(tuple(int(v) for v in board[:7]))

    if not X:
        return np.asarray([]), np.asarray([]), []
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int), contexts


def _split(
    X: np.ndarray,
    y: np.ndarray,
    contexts: list[tuple[int, ...]],
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[int, ...]], list[tuple[int, ...]]]:
    count = len(X)
    indices = list(range(count))
    rng = Random(random_state)
    rng.shuffle(indices)
    test_count = max(1, int(count * test_size))
    train_count = max(1, count - test_count)
    train_idx = indices[:train_count]
    test_idx = indices[train_count:]
    if not test_idx:
        test_idx = train_idx[-1:]
        train_idx = train_idx[:-1] or train_idx
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    ctx_train = [contexts[idx] for idx in train_idx]
    ctx_test = [contexts[idx] for idx in test_idx]
    return X_train, X_test, y_train, y_test, ctx_train, ctx_test


def _majority_baseline(y_true: np.ndarray, y_train: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    majority = int(np.argmax(np.bincount(y_train if len(y_train) else y_true)))
    return float(np.mean(y_true == majority))


def training_readiness(rows: list[dict], lookback: int, minimum_samples: int = 20) -> dict:
    """Summarize whether current move data can support supervised training."""

    X, _, _ = build_dataset(rows, lookback=lookback)
    sample_count = int(len(X))
    session_keys = {
        (int(row["game_id"]), int(row["session_index"]))
        for row in rows
        if "game_id" in row and "session_index" in row
    }
    return {
        "total_move_rows": int(len(rows)),
        "session_count": int(len(session_keys)),
        "lookback": int(lookback),
        "sample_count": sample_count,
        "minimum_required_samples": int(minimum_samples),
        "sample_formula": "Each session with n AI moves contributes max(0, n - lookback) samples.",
        "can_train": sample_count >= minimum_samples,
        "sklearn_available": bool(SKLEARN_AVAILABLE),
        "sklearn_import_error": SKLEARN_IMPORT_ERROR,
    }


def train_model(rows: list[dict], config: TrainConfig, artifact_path: str) -> dict[str, Any]:
    """Train one supervised model and persist artifact/metrics."""

    X, y, contexts = build_dataset(rows, lookback=config.lookback)
    if len(X) < 20:
        raise RuntimeError("Not enough training samples. Play more games or lower lookback.")
    X_train, X_test, y_train, y_test, ctx_train, ctx_test = _split(
        X,
        y,
        contexts,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    model_type = config.model_type
    if model_type == "decision_tree":
        if DecisionTreeClassifier is None:
            raise RuntimeError(
                "scikit-learn is required for decision_tree training"
                + (f". Import error: {SKLEARN_IMPORT_ERROR}" if SKLEARN_IMPORT_ERROR else "")
            )
        model = DecisionTreeClassifier(random_state=config.random_state)
        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        train_acc = float(accuracy_score(y_train, pred_train)) if accuracy_score else float(np.mean(pred_train == y_train))
        test_acc = float(accuracy_score(y_test, pred_test)) if accuracy_score else float(np.mean(pred_test == y_test))
    elif model_type == "mlp":
        if MLPClassifier is None:
            raise RuntimeError(
                "scikit-learn is required for mlp training"
                + (f". Import error: {SKLEARN_IMPORT_ERROR}" if SKLEARN_IMPORT_ERROR else "")
            )
        model = MLPClassifier(
            hidden_layer_sizes=config.hidden_layer_sizes,
            learning_rate_init=config.learning_rate,
            max_iter=config.epochs,
            batch_size=config.batch_size,
            random_state=config.random_state,
        )
        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        train_acc = float(accuracy_score(y_train, pred_train)) if accuracy_score else float(np.mean(pred_train == y_train))
        test_acc = float(accuracy_score(y_test, pred_test)) if accuracy_score else float(np.mean(pred_test == y_test))
    elif model_type == "frequency":
        model = FrequencyModel().fit(ctx_train, y_train)
        pred_train = model.predict_contexts(ctx_train)
        pred_test = model.predict_contexts(ctx_test)
        train_acc = float(np.mean(pred_train == y_train))
        test_acc = float(np.mean(pred_test == y_test))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    metrics = {
        "sample_count": int(len(X)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "baseline_accuracy": _majority_baseline(y_test, y_train),
        "lookback": int(config.lookback),
        "epochs": int(config.epochs),
        "batch_size": config.batch_size,
    }
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "config": asdict(config),
        "model_type": model_type,
        "model": model,
    }
    payload = pickle.dumps(artifact)
    write_bytes(artifact_path, payload, content_type="application/octet-stream")
    metrics["artifact_path"] = artifact_path
    return metrics


def load_artifact(path: str) -> dict[str, Any]:
    """Load serialized model artifact from local or object storage."""

    payload = read_bytes(path)
    return pickle.loads(payload)


def _feature_vector(board: list[int], history_actions: list[int], lookback: int) -> np.ndarray:
    features: list[float] = [float(v) for v in board]
    history = history_actions[-lookback:]
    if len(history) < lookback:
        history = [0] * (lookback - len(history)) + history
    for action in history:
        one_hot = [0.0] * 7
        if 0 <= int(action) < 7:
            one_hot[int(action)] = 1.0
        features.extend(one_hot)
    return np.asarray(features, dtype=float)


def predict_action(
    artifact: dict[str, Any],
    board: list[int],
    history_actions: list[int],
    valid_moves: list[int],
) -> int | None:
    """Predict the next move from artifact + board state."""

    if not valid_moves:
        return None
    config = artifact.get("config", {})
    lookback = int(config.get("lookback", 5))
    model_type = artifact.get("model_type")
    model = artifact.get("model")

    if model_type == "frequency":
        context = tuple(int(v) for v in board[:7])
        prediction = int(model.predict_context(context))
    else:
        vec = _feature_vector(board, history_actions, lookback=lookback)
        prediction = int(model.predict(np.asarray([vec]))[0])

    if prediction in valid_moves:
        return prediction
    return None
