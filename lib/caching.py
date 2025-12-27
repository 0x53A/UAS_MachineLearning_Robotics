from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import hashlib
import json
from pathlib import Path


@dataclass
class TrainingState:
    weights: List[np.ndarray]
    biases: List[np.ndarray]
    epoch: int
    loss_history: List[float]


@dataclass
class CheckpointConfig:
    data_hash: str
    enabled: bool = False
    cache_dir: str = ".net_cache"
    checkpoint_interval: int = 10
    overwrite: bool = False


def compute_data_hash(inputs: List[np.ndarray], targets: List[np.ndarray]) -> str:
    """Compute a hash of the training data for cache invalidation."""
    combined = np.concatenate([np.concatenate(inputs), np.concatenate(targets)])
    return hashlib.sha256(combined.tobytes()).hexdigest()[:16]


def compute_config_hash(
    layers: list,
    loss_func,
    seed: Optional[int],
    learning_rate: float,
    batch_size: Optional[int],
    data_hash: Optional[str],
) -> str:
    """Compute a hash of the network configuration for caching."""
    config_dict = {
        "n_inputs": len(layers[0].neurons[0].connections),
        "n_outputs": len(layers[-1].neurons),
        "n_hidden_layers": len(layers) - 1,
        "n_neurons_per_hidden": len(layers[0].neurons),
        "hidden_activation_hash": layers[0].activation_func.hash_expr(),
        "output_activation_hash": layers[-1].activation_func.hash_expr(),
        "loss_func_hash": loss_func.hash_expr(),
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "data_hash": data_hash or "no_data_hash",
        "seed": seed,
    }

    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def get_cache_path(cache_dir: str, config_hash: str) -> Path:
    """Get the cache directory path for a given config hash."""
    cache_path = Path(cache_dir) / f"net_{config_hash}"
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def save_checkpoint(
    cache_dir: str,
    config_hash: str,
    epoch: int,
    weights: List[np.ndarray],
    biases: List[np.ndarray],
    loss_history: List[float],
) -> None:
    """Save a training checkpoint to disk."""
    cache_path = get_cache_path(cache_dir, config_hash)
    checkpoint_path = cache_path / f"checkpoint_epoch_{epoch}.npz"
    latest_path = cache_path / "latest.npz"

    state = {
        "epoch": epoch,
        "loss_history": np.array(loss_history),
        "num_layers": len(weights),
    }

    for i, w in enumerate(weights):
        state[f"weights_{i}"] = w

    for i, b in enumerate(biases):
        state[f"biases_{i}"] = b

    np.savez(latest_path, **state)
    np.savez(checkpoint_path, **state)


def load_checkpoint(
    cache_dir: str,
    config_hash: str,
    target_epoch: Optional[int] = None,
) -> Optional[TrainingState]:
    """
    Load a training checkpoint from disk.

    Args:
        cache_dir: Directory where checkpoints are stored
        config_hash: Hash identifying the network configuration
        target_epoch: If specified, only return a checkpoint with at most this many epochs.
                     We can continue training from epoch N to target_epoch, but we can't
                     "untrain" from a checkpoint that has more epochs than requested.
                     If None, loads the latest checkpoint regardless of epoch count.

    Returns:
        TrainingState if a valid checkpoint is found, None otherwise.
    """
    cache_path = get_cache_path(cache_dir, config_hash)

    if target_epoch is not None:
        # Find the best checkpoint: highest epoch that doesn't exceed target_epoch
        best_checkpoint_path = None
        best_epoch = 0

        # Check all checkpoint files
        for checkpoint_file in cache_path.glob("checkpoint_epoch_*.npz"):
            try:
                epoch_str = checkpoint_file.stem.replace("checkpoint_epoch_", "")
                epoch = int(epoch_str)
                if epoch <= target_epoch and epoch > best_epoch:
                    best_epoch = epoch
                    best_checkpoint_path = checkpoint_file
            except ValueError:
                continue

        if best_checkpoint_path is None:
            return None

        checkpoint_path = best_checkpoint_path
    else:
        # No target epoch specified, use latest
        checkpoint_path = cache_path / "latest.npz"
        if not checkpoint_path.exists():
            return None

    try:
        data = np.load(checkpoint_path, allow_pickle=True)
        num_layers = int(data["num_layers"])
        checkpoint_epoch = int(data["epoch"])

        weights = [data[f"weights_{i}"] for i in range(num_layers)]
        biases = [data[f"biases_{i}"] for i in range(num_layers)]

        return TrainingState(
            weights=weights,
            biases=biases,
            epoch=checkpoint_epoch,
            loss_history=list(data["loss_history"]),
        )
    except Exception:
        return None
