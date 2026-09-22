"""Measurement backend for forward-query unitary learning.

This module is an exponential state-vector *simulator*, not part of the
polynomial classical learner.  Its public sampling interface supplies only
measurement outcomes.  The learner must not inspect the private Choi vector,
the target gate list, or the target matrix.

Choi registers are ordered ``reference, output``, with q0 most significant.
Each Bell record consumes two independent preparations of the same Choi state,
and hence two forward queries.  Each decoded Pauli shot consumes one forward
query.  The simulator caches the Choi vector and measurement probabilities to
avoid repeating identical numerical work; ``counts['forward_queries']`` counts
the queries that the represented experiment would consume, whereas
``simulator_forward_evaluations`` counts actual callback invocations.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence

import numpy as np

from verify_algorithm import GATES as _LEGACY_GATES
from verify_algorithm import apply_gate as _legacy_apply_gate
from verify_algorithm import bell_distribution


Gate = tuple
ForwardAction = Callable[[np.ndarray, tuple[int, ...]], np.ndarray]

_EXTRA_GATES = {
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.diag([1, -1]).astype(complex),
    'T': np.diag([1, np.exp(1j * np.pi / 4)]),
    'TDG': np.diag([1, np.exp(-1j * np.pi / 4)]),
}


def _qubit_count(state: np.ndarray) -> int:
    if state.ndim != 1 or state.size < 2 or state.size & (state.size - 1):
        raise ValueError('The state must be a one-dimensional qubit state vector.')
    return state.size.bit_length() - 1


def _checked_targets(targets, n: int) -> tuple[int, ...]:
    sites = tuple(operator.index(q) for q in targets)
    if len(set(sites)) != len(sites) or any(q < 0 or q >= n for q in sites):
        raise ValueError('Gate targets must be distinct valid qubit positions.')
    return sites


def _apply_matrix(state, matrix, targets):
    state = np.asarray(state, dtype=complex)
    n = _qubit_count(state)
    sites = _checked_targets(targets, n)
    matrix = np.asarray(matrix, dtype=complex)
    if matrix.shape != (2 ** len(sites), 2 ** len(sites)):
        raise ValueError('The matrix dimension does not match its targets.')
    order = list(sites) + [q for q in range(n) if q not in sites]
    tensor = state.reshape([2] * n).transpose(order).reshape(2 ** len(sites), -1)
    return (matrix @ tensor).reshape([2] * n).transpose(np.argsort(order)).ravel()


def apply_gate(state: np.ndarray, gate: Gate) -> np.ndarray:
    """Apply one gate using the package's basis and gate conventions."""
    state = np.asarray(state, dtype=complex)
    n = _qubit_count(state)
    if not gate:
        raise ValueError('An empty gate is not valid.')
    name, *targets = gate
    sites = _checked_targets(targets, n)
    table = _LEGACY_GATES if name in _LEGACY_GATES else _EXTRA_GATES
    if name not in table:
        raise ValueError(f'Unsupported gate {name!r}.')
    expected = table[name].shape[0].bit_length() - 1
    if len(sites) != expected:
        raise ValueError(f'{name} expects {expected} target qubits.')
    if name in _LEGACY_GATES:
        return _legacy_apply_gate(state, (name, *sites))
    return _apply_matrix(state, table[name], sites)


def apply_circuit(state: np.ndarray, circuit: Sequence[Gate]) -> np.ndarray:
    """Apply known gates; this utility is also used for external verification."""
    out = np.asarray(state, dtype=complex).copy()
    _qubit_count(out)
    for gate in circuit:
        out = apply_gate(out, gate)
    return out


def circuit_matrix(n: int, circuit: Sequence[Gate]) -> np.ndarray:
    """Construct a dense target matrix for small-system verification only."""
    n = operator.index(n)
    if n < 1:
        raise ValueError('n must be positive.')
    circuit = tuple(tuple(gate) for gate in circuit)
    return np.column_stack([apply_circuit(v, circuit) for v in np.eye(2 ** n)])


def choi_vector(n: int, apply_forward: ForwardAction) -> np.ndarray:
    """Prepare normalized ``2^-n/2 sum_x |x> U|x>`` with one forward action.

    ``apply_forward(joint_state, targets)`` must act as U on the listed qubits
    in that order, leaving all other qubits untouched.  It must support inputs
    entangled with a reference.  No inverse or controlled oracle is needed.
    This utility is simulator-side and is never exposed through the learner's
    measurement protocol.
    """
    n = operator.index(n)
    if n < 1:
        raise ValueError('n must be positive.')
    dimension = 2 ** n
    maximally_entangled = np.eye(dimension, dtype=complex).ravel() / np.sqrt(dimension)
    output = np.asarray(apply_forward(maximally_entangled, tuple(range(n, 2 * n))),
                        dtype=complex)
    if output.shape != maximally_entangled.shape or not np.all(np.isfinite(output)):
        raise ValueError('The forward action returned an invalid state vector.')
    if not np.isclose(np.vdot(output, output).real, 1.0, rtol=1e-10, atol=1e-10):
        raise ValueError('The forward action did not preserve the state norm.')
    return output.copy()


class StatevectorUnitaryOracle:
    """Forward-query measurement oracle simulated with state vectors.

    Default limit: n <= 5, because Bell sampling constructs a 4n-qubit vector.
    Increase ``max_n`` explicitly only after assessing memory requirements.
    A user-defined forward action is trusted to implement a fixed unitary;
    passing the Choi norm check alone does not certify that promise.
    """

    def __init__(self, n: int, apply_forward: ForwardAction, *, max_n: int = 5):
        self.n = operator.index(n)
        max_n = operator.index(max_n)
        if self.n < 1 or self.n > max_n:
            raise ValueError(f'State-vector simulation requires 1 <= n <= {max_n}.')
        if not callable(apply_forward):
            raise TypeError('apply_forward must be a callable.')
        self._apply_forward = apply_forward
        self._choi = None
        self._bell_probabilities = None
        self._decoded_cache = {}
        self.counts = {'bell_records': 0, 'decoded_pauli_shots': 0, 'forward_queries': 0}
        self.simulator_forward_evaluations = 0

    @classmethod
    def from_gates(cls, n: int, circuit: Sequence[Gate], *, max_n: int = 5):
        """Wrap a known target circuit, whose gates stay on the simulator side."""
        n = operator.index(n)
        gates = tuple(tuple(gate) for gate in circuit)
        for gate in gates:
            if not gate:
                raise ValueError('An empty gate is not valid.')
            _checked_targets(gate[1:], n)

        def forward(state, targets):
            if len(targets) != n:
                raise ValueError('The number of target qubits must equal n.')
            mapped = [(gate[0], *(targets[q] for q in gate[1:])) for gate in gates]
            return apply_circuit(state, mapped)

        return cls(n, forward, max_n=max_n)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray, *, max_n: int = 5):
        """Wrap a dense unitary; input and output axes follow q0-first order."""
        matrix = np.array(matrix, dtype=complex, copy=True)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError('The unitary matrix must be square.')
        dimension = matrix.shape[0]
        if dimension < 2 or dimension & (dimension - 1):
            raise ValueError('The unitary dimension must be a positive power of two.')
        n = dimension.bit_length() - 1
        if n > max_n:
            raise ValueError(f'State-vector simulation requires n <= {max_n}.')
        if not np.all(np.isfinite(matrix)) or not np.allclose(
                matrix.conj().T @ matrix, np.eye(dimension), rtol=1e-10, atol=1e-10):
            raise ValueError('The supplied matrix is not unitary.')

        def forward(state, targets):
            if len(targets) != n:
                raise ValueError('The number of target qubits must equal n.')
            return _apply_matrix(state, matrix, targets)

        return cls(n, forward, max_n=max_n)

    def _state(self):
        if self._choi is None:
            self._choi = choi_vector(self.n, self._apply_forward)
            self.simulator_forward_evaluations += 1
        return self._choi

    @staticmethod
    def _shots(shots):
        shots = operator.index(shots)
        if shots < 0:
            raise ValueError('The number of shots cannot be negative.')
        return shots

    def sample_bell(self, shots: int, rng: np.random.Generator) -> np.ndarray:
        """Sample Bell labels ``(x_0,...,x_(2n-1),z_0,...,z_(2n-1))``.

        The inherited Bell circuit uses two identical Choi states, never a
        complex-conjugate copy.  Only the outcome records are returned.
        """
        shots = self._shots(shots)
        if shots == 0:
            return np.empty((0, 4 * self.n), dtype=np.uint8)
        if self._bell_probabilities is None:
            self._bell_probabilities = bell_distribution(self._state())
        indices = rng.choice(len(self._bell_probabilities), size=shots,
                             p=self._bell_probabilities).astype(np.uint64)
        records = ((indices[:, None] >> np.arange(4 * self.n, dtype=np.uint64)) & 1)
        self.counts['bell_records'] += shots
        self.counts['forward_queries'] += 2 * shots
        return records.astype(np.uint8)

    def sample_decoded_paulis(self, decoder: Sequence[Gate], bases: Sequence[str],
                              shots: int, rng: np.random.Generator) -> np.ndarray:
        """Measure one Pauli per Choi qubit after a supplied known circuit.

        ``bases`` contains exactly 2n entries from X/Y/Z.  Columns are decoded
        qubits in physical order.  Outcomes within a shot are sampled jointly
        and retain their quantum correlations; rows are independent shots.
        The known decoder may include T/TDG when cancelling learned rotations.
        """
        shots = self._shots(shots)
        bases = tuple(bases)
        if len(bases) != 2 * self.n or any(b not in ('X', 'Y', 'Z') for b in bases):
            raise ValueError('bases must specify X, Y, or Z for every Choi qubit.')
        decoder = tuple(tuple(gate) for gate in decoder)
        if shots == 0:
            return np.empty((0, 2 * self.n), dtype=np.int8)
        if decoder not in self._decoded_cache:
            self._decoded_cache[decoder] = apply_circuit(self._state(), decoder)
        rotations = []
        for q, basis in enumerate(bases):
            if basis == 'Y':
                rotations.append(('SDG', q))
            if basis in ('X', 'Y'):
                rotations.append(('H', q))
        rotated = apply_circuit(self._decoded_cache[decoder], rotations)
        probabilities = np.abs(rotated) ** 2
        probabilities /= probabilities.sum()
        indices = rng.choice(len(probabilities), size=shots, p=probabilities)
        bits = (indices[:, None] >> np.arange(2 * self.n - 1, -1, -1)) & 1
        self.counts['decoded_pauli_shots'] += shots
        self.counts['forward_queries'] += shots
        return (1 - 2 * bits).astype(np.int8)
