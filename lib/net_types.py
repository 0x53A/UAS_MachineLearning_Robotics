from dataclasses import dataclass, field, asdict
from typing import List, Optional, Callable, Dict, Any
from enum import Enum
import numpy as np
import sympy as sp
import hashlib
import json
from pathlib import Path

from lib.activation import SymbolicActivation
from lib.caching import CheckpointConfig, TrainingState
from lib.loss import SymbolicLoss


@dataclass
class Connection:
    from_index: int
    weight: float
    fixed_weight: bool = False


@dataclass
class Neuron:
    bias: float
    fixed_bias: bool
    connections: List[Connection] = field(default_factory=list)


@dataclass
class Layer:
    activation_func: SymbolicActivation
    neurons: List[Neuron] = field(default_factory=list)


class WeightInit(Enum):
    XAVIER = "xavier"
    HE = "he"
    NORMAL = "normal"
    UNIFORM_0_01 = "uniform_0_01"
    UNIFORM_0_1 = "uniform_0_1"


@dataclass
class NetConfig:
    n_inputs: int
    n_outputs: int
    n_hidden_layers: int
    n_neurons_per_hidden: int
    hidden_activation: SymbolicActivation = field(
        default_factory=lambda: SymbolicActivation.relu()
    )
    output_activation: SymbolicActivation = field(
        default_factory=lambda: SymbolicActivation.linear()
    )
    loss_func: SymbolicLoss = field(
        default_factory=lambda: SymbolicLoss.cross_entropy()
    )
    weight_init: WeightInit = WeightInit.XAVIER
    bias_init_std: float = 0.0
    seed: int = 42


def compute_data_hash(inputs: List[np.ndarray], targets: List[np.ndarray]) -> str:
    combined = np.concatenate([np.concatenate(inputs), np.concatenate(targets)])
    return hashlib.sha256(combined.tobytes()).hexdigest()[:16]


@dataclass
class Net:
    layers: List[Layer]
    loss_func: SymbolicLoss
    seed: Optional[int] = None

    def __post_init__(self):
        self._config_hash: Optional[str] = None

    def _to_arrays(self):
        weights = []
        biases = []

        for layer in self.layers:
            layer_weights = []
            layer_biases = []

            for neuron in layer.neurons:
                layer_biases.append(neuron.bias)
                neuron_weights = [conn.weight for conn in neuron.connections]
                layer_weights.append(neuron_weights)

            weights.append(np.array(layer_weights))
            biases.append(np.array(layer_biases))

        return weights, biases

    def _from_arrays(self, weights: List[np.ndarray], biases: List[np.ndarray]):
        for layer_idx, layer in enumerate(self.layers):
            for neuron_idx, neuron in enumerate(layer.neurons):
                neuron.bias = biases[layer_idx][neuron_idx]
                for conn_idx, conn in enumerate(neuron.connections):
                    conn.weight = weights[layer_idx][neuron_idx][conn_idx]

    def forward(self, inputs: np.ndarray):
        weights, biases = self._to_arrays()
        activations = [inputs]
        pre_activations = []

        x = inputs
        for layer_idx, layer in enumerate(self.layers):
            activation = layer.activation_func
            W = weights[layer_idx]
            b = biases[layer_idx]

            z = np.dot(x, W.T) + b
            pre_activations.append(z)

            x = activation.forward_func(z)
            activations.append(x)

        return activations, pre_activations

    def forward_batch(self, inputs_batch: np.ndarray):
        """Forward pass for a batch of inputs."""
        weights, biases = self._to_arrays()
        activations = [inputs_batch]
        pre_activations = []

        x = inputs_batch
        for layer_idx, layer in enumerate(self.layers):
            activation = layer.activation_func
            W = weights[layer_idx]
            b = biases[layer_idx]

            z = np.dot(x, W.T) + b
            pre_activations.append(z)

            x = activation.forward_func(z)
            activations.append(x)

        return activations, pre_activations

    def backward(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        activations: List[np.ndarray],
        pre_activations: List[np.ndarray],
    ):
        weights, _ = self._to_arrays()
        d_weights = [np.zeros_like(w) for w in weights]
        d_biases = [np.zeros_like(b) for _, b in enumerate(weights)]

        loss = self.loss_func
        delta = loss.grad_func(y_pred, y_true)

        for layer_idx in reversed(range(len(self.layers))):
            activation = self.layers[layer_idx].activation_func
            a_prev = activations[layer_idx]
            z = pre_activations[layer_idx]

            if not activation.is_softmax():
                delta = delta * activation.deriv_func(z)

            d_weights[layer_idx] = np.outer(delta, a_prev)
            d_biases[layer_idx] = delta

            if layer_idx > 0:
                W = weights[layer_idx]
                delta = np.dot(delta, W)

        return d_weights, d_biases

    def backward_batch(
        self,
        y_pred_batch: np.ndarray,
        y_true_batch: np.ndarray,
        activations: List[np.ndarray],
        pre_activations: List[np.ndarray],
    ):
        """Backward pass for a batch of samples."""
        weights, _ = self._to_arrays()
        batch_size = y_pred_batch.shape[0]
        d_weights = [np.zeros_like(w) for w in weights]
        d_biases = [np.zeros_like(b) for _, b in enumerate(weights)]

        loss = self.loss_func
        delta_batch = loss.grad_func(y_pred_batch, y_true_batch)

        for layer_idx in reversed(range(len(self.layers))):
            activation = self.layers[layer_idx].activation_func
            a_prev = activations[layer_idx]
            z_batch = pre_activations[layer_idx]

            if not activation.is_softmax():
                delta_batch = delta_batch * activation.deriv_func(z_batch)

            d_weights[layer_idx] = np.sum(
                np.stack(
                    [np.outer(delta_batch[i], a_prev[i]) for i in range(batch_size)]
                ),
                axis=0,
            )
            d_biases[layer_idx] = np.sum(delta_batch, axis=0)

            if layer_idx > 0:
                W = weights[layer_idx]
                delta_batch = np.dot(delta_batch, W)

        return d_weights, d_biases

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        loss = self.loss_func
        result = loss.loss_func(y_pred, y_true)
        if isinstance(result, np.ndarray):
            return float(np.mean(result))
        return float(result)

    def compute_loss_batch(
        self, y_pred_batch: np.ndarray, y_true_batch: np.ndarray
    ) -> np.ndarray:
        """Compute loss for a batch of samples."""
        loss = self.loss_func
        result = loss.loss_func(y_pred_batch, y_true_batch)
        if isinstance(result, np.ndarray):
            return np.mean(result, axis=1)
        return result

    def _compute_config_hash(
        self,
        learning_rate: float,
        batch_size: Optional[int],
        data_hash: Optional[str],
    ):
        config_dict = {
            "n_inputs": len(self.layers[0].neurons[0].connections),
            "n_outputs": len(self.layers[-1].neurons),
            "n_hidden_layers": len(self.layers) - 1,
            "n_neurons_per_hidden": len(self.layers[0].neurons),
            "hidden_activation_hash": self.layers[0].activation_func.hash_expr(),
            "output_activation_hash": self.layers[-1].activation_func.hash_expr(),
            "loss_func_hash": self.loss_func.hash_expr(),
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "data_hash": data_hash or "no_data_hash",
            "seed": self.seed,
        }

        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _get_cache_path(self, cache_dir: str, config_hash: str) -> Path:
        cache_path = Path(cache_dir) / f"net_{config_hash}"
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path

    def _save_checkpoint(
        self,
        cache_dir: str,
        config_hash: str,
        epoch: int,
        weights: List[np.ndarray],
        biases: List[np.ndarray],
        loss_history: List[float],
    ):
        cache_path = self._get_cache_path(cache_dir, config_hash)
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

    def _load_checkpoint(
        self,
        cache_dir: str,
        config_hash: str,
    ) -> Optional[TrainingState]:
        cache_path = self._get_cache_path(cache_dir, config_hash)
        latest_path = cache_path / "latest.npz"

        if not latest_path.exists():
            return None

        try:
            data = np.load(latest_path, allow_pickle=True)
            num_layers = int(data["num_layers"])

            weights = [data[f"weights_{i}"] for i in range(num_layers)]
            biases = [data[f"biases_{i}"] for i in range(num_layers)]

            return TrainingState(
                weights=weights,
                biases=biases,
                epoch=int(data["epoch"]),
                loss_history=list(data["loss_history"]),
            )
        except Exception:
            return None

    def _train_sample_by_sample(
        self,
        inputs: List[np.ndarray],
        targets: List[np.ndarray],
        learning_rate: float,
    ) -> float:
        """Train for one epoch, processing samples one at a time."""
        n_samples = len(inputs)
        epoch_loss = 0.0

        for i in range(n_samples):
            activations, pre_activations = self.forward(inputs[i])
            y_pred = activations[-1]
            loss = self.compute_loss(y_pred, targets[i])
            epoch_loss += loss

            d_weights, d_biases = self.backward(
                y_pred, targets[i], activations, pre_activations
            )

            weights, biases = self._to_arrays()
            for layer_idx in range(len(weights)):
                for neuron_idx in range(len(weights[layer_idx])):
                    biases[layer_idx][neuron_idx] -= (
                        learning_rate * d_biases[layer_idx][neuron_idx]
                    )
                    for conn_idx in range(len(weights[layer_idx][neuron_idx])):
                        weights[layer_idx][neuron_idx][conn_idx] -= (
                            learning_rate * d_weights[layer_idx][neuron_idx][conn_idx]
                        )

            self._from_arrays(weights, biases)

        return epoch_loss

    def _train_batch(
        self,
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
            batch_targets = np.array(
                [targets[i] for i in range(batch_start, batch_end)]
            )

            activations, pre_activations = self.forward_batch(batch_inputs)
            y_pred_batch = activations[-1]
            loss_batch = self.compute_loss_batch(y_pred_batch, batch_targets)
            epoch_loss += np.sum(loss_batch)

            d_weights, d_biases = self.backward_batch(
                y_pred_batch, batch_targets, activations, pre_activations
            )

            batch_size_actual = batch_end - batch_start
            weights, biases = self._to_arrays()
            for layer_idx in range(len(weights)):
                biases[layer_idx] -= (learning_rate / batch_size_actual) * d_biases[
                    layer_idx
                ]
                weights[layer_idx] -= (learning_rate / batch_size_actual) * d_weights[
                    layer_idx
                ]

            self._from_arrays(weights, biases)

        return epoch_loss

    def train(
        self,
        inputs: List[np.ndarray],
        targets: List[np.ndarray],
        learning_rate: float,
        epochs: int,
        batch_size: Optional[int] = None,
        checkpoint_config: Optional[CheckpointConfig] = None,
    ) -> Dict[str, Any]:
        n_samples = len(inputs)
        loss_history: List[float] = []
        start_epoch = 0
        config_hash = None

        # Determine if we can use batch processing
        # If the loss function doesn't support batching, we fall back to sample-by-sample
        use_batch = (
            batch_size is not None
            and batch_size < n_samples
            and self.loss_func.supports_batch
        )

        if checkpoint_config and checkpoint_config.enabled:
            config_hash = self._compute_config_hash(
                learning_rate, batch_size, checkpoint_config.data_hash
            )
            self._config_hash = config_hash

            loaded_state = self._load_checkpoint(
                checkpoint_config.cache_dir, config_hash
            )
            if loaded_state and not checkpoint_config.overwrite:
                # Only continue from checkpoint if we haven't already trained enough
                if loaded_state.epoch < epochs:
                    start_epoch = loaded_state.epoch
                    loss_history = loaded_state.loss_history[:start_epoch]
                    self._from_arrays(loaded_state.weights, loaded_state.biases)
                    print(
                        f"Loaded checkpoint from epoch {start_epoch}, "
                        f"continuing to epoch {epochs}..."
                    )
                else:
                    # Checkpoint has enough or more epochs than requested
                    # Load the weights and truncate loss history to requested epochs
                    self._from_arrays(loaded_state.weights, loaded_state.biases)
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
                epoch_loss = self._train_batch(
                    inputs, targets, learning_rate, batch_size
                )
            else:
                epoch_loss = self._train_sample_by_sample(
                    inputs, targets, learning_rate
                )

            avg_loss = epoch_loss / n_samples
            loss_history.append(avg_loss)

            if (
                checkpoint_config
                and checkpoint_config.enabled
                and config_hash is not None
                and (epoch + 1) % checkpoint_config.checkpoint_interval == 0
            ):
                weights, biases = self._to_arrays()
                self._save_checkpoint(
                    checkpoint_config.cache_dir,
                    config_hash,
                    epoch + 1,
                    weights,
                    biases,
                    loss_history,
                )
                print(f"Saved checkpoint at epoch {epoch + 1}")

        return {"loss_history": loss_history}

    @staticmethod
    def fully_connected(config: "NetConfig") -> "Net":
        """Create a fully connected neural network from config."""
        rng = np.random.RandomState(config.seed)
        layers: List[Layer] = []

        layer_sizes = [config.n_inputs]
        layer_sizes.extend([config.n_neurons_per_hidden] * config.n_hidden_layers)
        layer_sizes.append(config.n_outputs)

        for layer_idx in range(1, len(layer_sizes)):
            input_size = layer_sizes[layer_idx - 1]
            output_size = layer_sizes[layer_idx]

            if layer_idx == len(layer_sizes) - 1:
                activation = config.output_activation
            else:
                activation = config.hidden_activation

            neurons: List[Neuron] = []

            for neuron_idx in range(output_size):
                bias = (
                    rng.normal(0, config.bias_init_std)
                    if config.bias_init_std > 0
                    else 0.0
                )

                connections: List[Connection] = []
                for input_idx in range(input_size):
                    if config.weight_init == WeightInit.XAVIER:
                        limit = np.sqrt(6.0 / (input_size + output_size))
                        weight = rng.uniform(-limit, limit)
                    elif config.weight_init == WeightInit.HE:
                        std = np.sqrt(2.0 / input_size)
                        weight = rng.normal(0, std)
                    elif config.weight_init == WeightInit.NORMAL:
                        weight = rng.normal(0, 0.1)
                    elif config.weight_init == WeightInit.UNIFORM_0_01:
                        weight = rng.uniform(-0.1, 0.1)
                    elif config.weight_init == WeightInit.UNIFORM_0_1:
                        weight = rng.uniform(-0.1, 1.0)
                    else:
                        weight = rng.normal(0, 0.1)

                    connections.append(
                        Connection(
                            from_index=input_idx, weight=weight, fixed_weight=False
                        )
                    )

                neurons.append(
                    Neuron(bias=bias, fixed_bias=False, connections=connections)
                )

            layers.append(Layer(activation_func=activation, neurons=neurons))

        net = Net(layers=layers, loss_func=config.loss_func)
        net.seed = config.seed
        return net
