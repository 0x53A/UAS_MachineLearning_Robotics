from typing import List, Optional, Dict, Any, TYPE_CHECKING
import numpy as np

from lib.caching import (
    CheckpointConfig,
    TrainingState,
    compute_config_hash,
    save_checkpoint,
    load_checkpoint,
)

if TYPE_CHECKING:
    from lib.net_types import Net


def _train_sample_by_sample(
    net: "Net",
    inputs: List[np.ndarray],
    targets: List[np.ndarray],
    learning_rate: float,
) -> float:
    """Train for one epoch, processing samples one at a time."""
    n_samples = len(inputs)
    epoch_loss = 0.0

    for i in range(n_samples):
        activations, pre_activations = net.forward(inputs[i])
        y_pred = activations[-1]
        loss = net.compute_loss(y_pred, targets[i])
        epoch_loss += loss

        d_weights, d_biases = net.backward(
            y_pred, targets[i], activations, pre_activations
        )

        weights, biases = net._to_arrays()
        for layer_idx in range(len(weights)):
            for neuron_idx in range(len(weights[layer_idx])):
                biases[layer_idx][neuron_idx] -= (
                    learning_rate * d_biases[layer_idx][neuron_idx]
                )
                for conn_idx in range(len(weights[layer_idx][neuron_idx])):
                    weights[layer_idx][neuron_idx][conn_idx] -= (
                        learning_rate * d_weights[layer_idx][neuron_idx][conn_idx]
                    )

        net._from_arrays(weights, biases)

    return epoch_loss


def _train_batch(
    net: "Net",
    inputs: List[np.ndarray],
    targets: List[np.ndarray],
    learning_rate: float,
    batch_size: int,
) -> float:
    """Train for one epoch using mini-batch processing."""
    n_samples = len(inputs)
    epoch_loss = 0.0

    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        batch_inputs = np.array([inputs[i] for i in range(batch_start, batch_end)])
        batch_targets = np.array([targets[i] for i in range(batch_start, batch_end)])

        activations, pre_activations = net.forward_batch(batch_inputs)
        y_pred_batch = activations[-1]
        loss_batch = net.compute_loss_batch(y_pred_batch, batch_targets)
        epoch_loss += np.sum(loss_batch)

        d_weights, d_biases = net.backward_batch(
            y_pred_batch, batch_targets, activations, pre_activations
        )

        batch_size_actual = batch_end - batch_start
        weights, biases = net._to_arrays()
        for layer_idx in range(len(weights)):
            biases[layer_idx] -= (learning_rate / batch_size_actual) * d_biases[
                layer_idx
            ]
            weights[layer_idx] -= (learning_rate / batch_size_actual) * d_weights[
                layer_idx
            ]

        net._from_arrays(weights, biases)

    return epoch_loss


def train(
    net: "Net",
    inputs: List[np.ndarray],
    targets: List[np.ndarray],
    learning_rate: float,
    epochs: int,
    batch_size: Optional[int] = None,
    checkpoint_config: Optional[CheckpointConfig] = None,
) -> Dict[str, Any]:
    """
    Train the neural network.

    Args:
        net: The neural network to train
        inputs: List of input arrays
        targets: List of target arrays
        learning_rate: Learning rate for gradient descent
        epochs: Number of epochs to train
        batch_size: Optional batch size for mini-batch training
        checkpoint_config: Optional checkpoint configuration for saving/loading state

    Returns:
        Dictionary containing training results including loss_history
    """
    n_samples = len(inputs)
    loss_history: List[float] = []
    start_epoch = 0
    config_hash = None

    # Determine if we can use batch processing
    # If the loss function doesn't support batching, we fall back to sample-by-sample
    use_batch = (
        batch_size is not None
        and batch_size < n_samples
        and net.loss_func.supports_batch
    )

    if checkpoint_config and checkpoint_config.enabled:
        config_hash = compute_config_hash(
            net.layers,
            net.loss_func,
            net.seed,
            learning_rate,
            batch_size,
            checkpoint_config.data_hash,
        )
        net._config_hash = config_hash

        # Load checkpoint, but only if it has at least as many epochs as we need
        # to continue from (or we want to return early if it has enough)
        loaded_state = load_checkpoint(checkpoint_config.cache_dir, config_hash)
        if loaded_state and not checkpoint_config.overwrite:
            # Only continue from checkpoint if we haven't already trained enough
            if loaded_state.epoch < epochs:
                start_epoch = loaded_state.epoch
                loss_history = loaded_state.loss_history[:start_epoch]
                net._from_arrays(loaded_state.weights, loaded_state.biases)
                print(
                    f"Loaded checkpoint from epoch {start_epoch}, "
                    f"continuing to epoch {epochs}..."
                )
            else:
                # Checkpoint has enough or more epochs than requested
                # Load the weights and truncate loss history to requested epochs
                net._from_arrays(loaded_state.weights, loaded_state.biases)
                loss_history = loaded_state.loss_history[:epochs]
                print(
                    f"Loaded checkpoint from epoch {loaded_state.epoch}, "
                    f"returning first {epochs} epochs of loss history."
                )
                return {"loss_history": loss_history}
        elif loaded_state and checkpoint_config.overwrite:
            print("Checkpoint found but overwrite=True, training from scratch...")
        else:
            start_epoch = 0

    for epoch in range(start_epoch, epochs):
        if use_batch and batch_size is not None:
            epoch_loss = _train_batch(net, inputs, targets, learning_rate, batch_size)
        else:
            epoch_loss = _train_sample_by_sample(net, inputs, targets, learning_rate)

        avg_loss = epoch_loss / n_samples
        loss_history.append(avg_loss)

        if (
            checkpoint_config
            and checkpoint_config.enabled
            and config_hash is not None
            and (epoch + 1) % checkpoint_config.checkpoint_interval == 0
        ):
            weights, biases = net._to_arrays()
            save_checkpoint(
                checkpoint_config.cache_dir,
                config_hash,
                epoch + 1,
                weights,
                biases,
                loss_history,
            )
            print(f"Saved checkpoint at epoch {epoch + 1}")

    return {"loss_history": loss_history}
