from dataclasses import dataclass, field, asdict
from typing import List, Optional, Callable, Dict, Any
from enum import Enum
import numpy as np
import sympy as sp
import hashlib
import json
from pathlib import Path




@dataclass
class SymbolicActivation:
    """Symbolic activation function with compiled forward and derivative."""

    expression: Any
    derivative: Any
    forward_func: Callable[[np.ndarray], np.ndarray]
    deriv_func: Callable[[np.ndarray], np.ndarray]
    name: str = "custom"

    @staticmethod
    def _make(
        z_sym: sp.Symbol,
        expr: Any,
        name: str,
    ):
        deriv = sp.diff(expr, z_sym)
        return SymbolicActivation(
            expression=expr,
            derivative=deriv,
            forward_func=sp.lambdify(z_sym, expr, "numpy"),
            deriv_func=sp.lambdify(z_sym, deriv, "numpy"),
            name=name,
        )

    @staticmethod
    def linear():
        z = sp.Symbol("z")
        return SymbolicActivation._make(z, z, "linear")

    @staticmethod
    def relu():
        z = sp.Symbol("z")
        expr = sp.Max(0, z)
        deriv = sp.Piecewise((0, z <= 0), (1, True))
        return SymbolicActivation(
            expression=expr,
            derivative=deriv,
            forward_func=sp.lambdify(z, expr, "numpy"),
            deriv_func=sp.lambdify(z, deriv, "numpy"),
            name="relu",
        )

    @staticmethod
    def tanh():
        z = sp.Symbol("z")
        return SymbolicActivation._make(z, sp.tanh(z), "tanh")

    @staticmethod
    def sigmoid():
        z = sp.Symbol("z")
        expr = 1 / (1 + sp.exp(-z))
        return SymbolicActivation._make(z, expr, "sigmoid")

    @staticmethod
    def softmax():
        """Softmax activation (special case, not purely sympy-based)."""

        def forward_func(z: np.ndarray) -> np.ndarray:
            exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
            return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

        def deriv_func(z: np.ndarray) -> np.ndarray:
            raise NotImplementedError(
                "Softmax derivative is handled in loss function when using cross-entropy"
            )

        return SymbolicActivation(
            expression=sp.Symbol("softmax_special"),
            derivative=sp.Symbol("handled_in_loss"),
            forward_func=forward_func,
            deriv_func=deriv_func,
            name="softmax",
        )

    @staticmethod
    def custom(expression: str, input_var: str = "z", name: str = "custom"):
        z = sp.Symbol(input_var)
        expr = sp.sympify(expression)
        deriv = sp.diff(expr, z)
        deriv = deriv.doit() if hasattr(deriv, "doit") else deriv
        deriv = sp.simplify(deriv) if hasattr(deriv, "simplify") else deriv
        return SymbolicActivation(
            expression=expr,
            derivative=deriv,
            forward_func=sp.lambdify(z, expr, "numpy"),
            deriv_func=sp.lambdify(z, deriv, "numpy"),
            name=name,
        )

    def is_softmax(self):
        """Check if this is the softmax activation."""
        return self.name == "softmax"

    def hash_expr(self):
        """Get a hash of the symbolic expression."""
        return hashlib.sha256(str(self.expression).encode()).hexdigest()[:16]

