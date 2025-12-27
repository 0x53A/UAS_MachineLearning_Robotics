from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from enum import Enum
import numpy as np

from lib.activation import SymbolicActivation
from lib.loss import SymbolicLoss

if TYPE_CHECKING:
    from lib.caching import CheckpointConfig


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

    def train(
        self,
        inputs: List[np.ndarray],
        targets: List[np.ndarray],
        learning_rate: float,
        epochs: int,
        batch_size: Optional[int] = None,
        checkpoint_config: Optional["CheckpointConfig"] = None,
    ) -> Dict[str, Any]:
        """
        Train the neural network.

        This is a convenience method that delegates to lib.training.train().
        """
        from lib.training import train as train_fn

        return train_fn(
            self,
            inputs,
            targets,
            learning_rate,
            epochs,
            batch_size,
            checkpoint_config,
        )
