from dataclasses import dataclass, field, asdict
from typing import List, Optional, Callable, Dict, Any
from enum import Enum
import numpy as np
import sympy as sp
import hashlib
import json
from pathlib import Path


@dataclass
class SymbolicLoss:
    """Symbolic loss function with compiled value and gradient.

    Attributes:
        expression: The symbolic expression for the loss.
        gradient: The symbolic gradient expression.
        loss_func: Compiled function to compute loss.
        grad_func: Compiled function to compute gradient.
        name: Name of the loss function.
        supports_batch: If True, the loss/gradient functions support batch processing.
                       If False, training will process samples one at a time.
    """

    expression: Any
    gradient: Any
    loss_func: Callable[[np.ndarray, np.ndarray], np.ndarray]
    grad_func: Callable[[np.ndarray, np.ndarray], np.ndarray]
    name: str = "custom"
    supports_batch: bool = True

    @staticmethod
    def _make(
        input_sym: sp.Symbol,
        target_sym: sp.Symbol,
        expr: Any,
        name: str,
        supports_batch: bool = True,
    ):
        deriv = sp.diff(expr, input_sym)
        deriv = deriv.doit() if hasattr(deriv, "doit") else deriv
        deriv = sp.simplify(deriv) if hasattr(deriv, "simplify") else deriv
        return SymbolicLoss(
            expression=expr,
            gradient=deriv,
            loss_func=sp.lambdify((input_sym, target_sym), expr, "numpy"),
            grad_func=sp.lambdify((input_sym, target_sym), deriv, "numpy"),
            name=name,
            supports_batch=supports_batch,
        )

    @staticmethod
    def mse():
        y_pred = sp.Symbol("y_pred")
        y_true = sp.Symbol("y_true")
        expr = (y_pred - y_true) ** 2
        return SymbolicLoss._make(y_pred, y_true, expr, "mse")

    @staticmethod
    def cross_entropy():
        """Cross-entropy loss (optimized for use with softmax)."""
        y_pred = sp.Symbol("y_pred")
        y_true = sp.Symbol("y_true")

        def loss_func(y_p: np.ndarray, y_t: np.ndarray) -> np.ndarray:
            return -np.sum(y_t * np.log(y_p + 1e-15))

        def grad_func(y_p: np.ndarray, y_t: np.ndarray) -> np.ndarray:
            return y_p - y_t

        return SymbolicLoss(
            expression=sp.Symbol("cross_entropy_optimized"),
            gradient=sp.Symbol("y_pred - y_true"),
            loss_func=loss_func,
            grad_func=grad_func,
            name="cross_entropy",
        )

    @staticmethod
    def binary_cross_entropy():
        y_pred = sp.Symbol("y_pred")
        y_true = sp.Symbol("y_true")
        expr = -(y_true * sp.log(y_pred) + (1 - y_true) * sp.log(1 - y_pred))
        return SymbolicLoss._make(y_pred, y_true, expr, "binary_cross_entropy")

    @staticmethod
    def custom(
        expression: str,
        pred_var: str = "y_pred",
        target_var: str = "y_true",
        name: str = "custom",
        supports_batch: bool = True,
    ):
        """Create a custom symbolic loss function.

        Args:
            expression: A sympy-compatible expression string.
            pred_var: Name of the prediction variable in the expression.
            target_var: Name of the target variable in the expression.
            name: Name for this loss function.
            supports_batch: If True, the loss supports batch training. Set to False
                           for losses that don't vectorize properly (e.g., those using
                           Abs, Piecewise, etc.).
        """
        # Create symbols with real=True assumption for proper differentiation
        y_pred = sp.Symbol(pred_var, real=True)
        y_true = sp.Symbol(target_var, real=True)

        # Parse the expression, then substitute in our properly-typed symbols
        # This ensures differentiation works correctly (e.g., for Abs)
        expr_parsed = sp.sympify(expression)
        old_pred = sp.Symbol(pred_var)  # Symbol without assumptions from sympify
        old_true = sp.Symbol(target_var)
        expr = expr_parsed.subs([(old_pred, y_pred), (old_true, y_true)])

        return SymbolicLoss._make(y_pred, y_true, expr, name, supports_batch)

    def hash_expr(self):
        """Get a hash of the symbolic expression."""
        return hashlib.sha256(str(self.expression).encode()).hexdigest()[:16]
