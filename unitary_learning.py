"""Forward-query learning of ancilla-free T-depth-one unitaries.

Promise: U = C_out (T**tensor(t) tensor I) C_in, 0 <= t <= n.
The learner receives only Bell records and sampled single-qubit Pauli outcomes
on decoded Choi copies. It never accesses a state vector or the target circuit.
Gate lists are chronological; qubit zero is the first tensor factor.
"""
from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Protocol

import numpy as np

from clifford_learning import (InsufficientData, Recovery, mm, rank,
                               recover_cuts, symplectic_form)
from signed_clifford import (conjugate_paulis, extend_commuting_axes,
                            inverse_circuit, multiply_paulis, solve_binary,
                            synthesize_clifford)


class ChoiMeasurementOracle(Protocol):
    """Measurement interface; no inverse or controlled-U operation is needed.

    Choi qubits are R_0,...,R_(n-1),A_0,...,A_(n-1). Bell labels contain all
    2n x bits followed by all 2n z bits. Each Bell row costs two forward calls.
    Each decoded Pauli shot costs one forward call and returns 2n signs.
    """
    n: int

    def sample_bell(self, shots: int, rng: np.random.Generator) -> np.ndarray: ...

    def sample_decoded_paulis(self, decoder: list, bases: list[str], shots: int,
                             rng: np.random.Generator) -> np.ndarray: ...


LOCAL_TOLERANCE = 1 / (4 * math.sqrt(2))


def sample_budgets(n, delta=0.01):
    """Conservative sufficient budgets from gamma_2 >= 1/16 and Hoeffding.

    Union bound over all Boolean quadratics on 4n Bell bits; half the failure
    budget is used for structural recovery and half for 3*(2n) local means.
    Returns (Bell records, fresh Choi shots for EACH of the X/Y/Z settings).
    """
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError('n must be a positive integer')
    if not 0 < delta < 1:
        raise ValueError('delta must lie strictly between zero and one')
    d = 4 * int(n)
    feature_count = 1 + d + d * (d - 1) // 2
    bell = math.ceil(16 * (feature_count * math.log(2) + math.log(2 / delta)))
    pauli = math.ceil(2 / LOCAL_TOLERANCE**2 * math.log(24 * n / delta))
    return bell, pauli


@lru_cache(maxsize=1)
def _t_orbit():
    """Twelve Bloch vectors and Clifford preparations, enumerating only Cl_1.

    This constant-size enumeration identifies local orientations. It is not a
    search over the unknown n-qubit Clifford layers.
    """
    xyz = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.uint8)
    seed = np.array([1., 1., 0.]) / math.sqrt(2)
    seen, orbit, queue = set(), {}, [[]]
    while queue:
        gates = queue.pop(0)
        rows, signs = conjugate_paulis(xyz, np.ones(3, dtype=int), gates)
        key = (rows.tobytes(), tuple(int(s) for s in signs))
        if key in seen:
            continue
        seen.add(key)
        vector = np.zeros(3)
        for source, row in enumerate(rows):
            dest = next(k for k, axis in enumerate(xyz) if np.array_equal(row, axis))
            vector[dest] = signs[source] * seed[source]
        orbit.setdefault(tuple(np.rint(vector * math.sqrt(2)).astype(int)),
                         (vector, tuple(gates)))
        queue.extend([gates + [('H', 0)], gates + [('S', 0)]])
    if len(seen) != 24 or len(orbit) != 12:
        raise RuntimeError('Internal single-qubit Clifford orbit construction failed')
    return tuple(orbit.values())


def _product(labels, signs, coefficients):
    label = np.zeros(labels.shape[1], dtype=np.uint8)
    sign = 1
    for i in np.flatnonzero(coefficients):
        label, sign = multiply_paulis(label, sign, labels[i], int(signs[i]))
    return label, sign


@dataclass
class UnitaryRecovery:
    n: int
    t: int
    input_clifford: list
    output_clifford: list
    remaining_clifford: list
    rotation_axes: np.ndarray
    rotation_signs: np.ndarray
    choi_preparation: list
    structure: Recovery
    bell_records: int
    pauli_shots_per_basis: int
    requested_delta: float
    failure_bound: float | None

    @property
    def circuit(self):
        """Chronological gates for C_out T_layer C_in, equal to U up to phase."""
        return self.input_clifford + [('T', q) for q in range(self.t)] + self.output_clifford

    @property
    def forward_queries(self):
        return 2 * self.bell_records + 3 * self.pauli_shots_per_basis

    def to_dict(self):
        return {
            'n': self.n, 't': self.t,
            'input_clifford': self.input_clifford,
            'output_clifford': self.output_clifford,
            'circuit': self.circuit,
            'remaining_clifford': self.remaining_clifford,
            'rotation_axes': self.rotation_axes.tolist(),
            'rotation_signs': self.rotation_signs.tolist(),
            'choi_preparation': self.choi_preparation,
            'choi_decoder': self.structure.decoder,
            'choi_stabilizer_rank': self.structure.stabilizer_rank,
            'bell_records': self.bell_records,
            'pauli_shots_per_basis': self.pauli_shots_per_basis,
            'forward_queries': self.forward_queries,
            'requested_delta': self.requested_delta,
            'failure_bound_under_exact_promise': self.failure_bound,
        }


def learn_t_depth_one(oracle: ChoiMeasurementOracle, *, delta=0.01, rng=None,
                      bell_shots=None, pauli_shots=None):
    """Recover a T-depth-one circuit using sampled measurements only.

    Default budgets suffice with success probability at least 1-delta under
    the exact promise. Optional smaller budgets are exploratory: the returned
    failure_bound is then None. InsufficientData means reconstruction failed;
    a returned hypothesis is not a membership certificate outside the promise.
    Stabilizer signs and local magic orientations use additional Pauli shots.
    """
    n = oracle.n
    default_bell, default_pauli = sample_budgets(n, delta)
    n, N = int(n), 2 * int(n)
    bell_shots = default_bell if bell_shots is None else bell_shots
    pauli_shots = default_pauli if pauli_shots is None else pauli_shots
    for value in (bell_shots, pauli_shots):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError('Shot counts must be positive integers')
    bell_shots, pauli_shots = int(bell_shots), int(pauli_shots)
    rng = np.random.default_rng(rng)
    records = np.asarray(oracle.sample_bell(bell_shots, rng))
    if records.shape != (bell_shots, 2 * N):
        raise ValueError('Oracle returned Bell records with the wrong shape')
    structure = recover_cuts(records)
    s, t = structure.stabilizer_rank, structure.nullity
    if t > n or any(len(block) != 1 for block in structure.blocks):
        raise InsufficientData('Recovered Choi structure is incompatible with T depth one')

    means = np.empty((3, N))
    for k, basis in enumerate(('X', 'Y', 'Z')):
        outcomes = np.asarray(oracle.sample_decoded_paulis(
            structure.decoder, [basis] * N, pauli_shots, rng))
        if outcomes.shape != (pauli_shots, N) or not np.all((outcomes == -1) | (outcomes == 1)):
            raise ValueError('Oracle must return a shots-by-2n array of +/-1 Pauli outcomes')
        means[k] = outcomes.mean(axis=0)

    # G maps |0>^s |T>^t to the Choi state. The decoder places fixed
    # qubits first, so there is no need to permute into a magic-first convention.
    local_preparation = []
    for q in range(s):
        target = np.array([0., 0., 1. if means[2, q] >= 0 else -1.])
        if np.max(np.abs(means[:, q] - target)) > LOCAL_TOLERANCE:
            raise InsufficientData('Decoded fixed-qubit measurements are inconsistent')
        if target[2] < 0:
            local_preparation.append(('X', q))
    for q in range(s, N):
        vector, gates = min(_t_orbit(), key=lambda item: np.linalg.norm(means[:, q] - item[0]))
        if np.max(np.abs(means[:, q] - vector)) > LOCAL_TOLERANCE:
            raise InsufficientData('Decoded magic-qubit measurements do not resolve the T orbit')
        local_preparation.extend((g[0], q) for g in gates)
    G = local_preparation + inverse_circuit(structure.decoder)

    # Learn signed stabilizers S_i=G Z_i G† and magic rotation axes G Z_j G†.
    z_labels = np.column_stack([np.zeros((N, N), np.uint8), np.eye(N, dtype=np.uint8)])
    gz, gz_signs = conjugate_paulis(z_labels, np.ones(N, dtype=int), G)
    stab, stab_signs = gz[:s], gz_signs[:s]
    reference = list(range(n)) + list(range(N, N + n))
    output = list(range(n, N)) + list(range(N + n, 2 * N))
    reference_stab = stab[:, reference].T
    if rank(reference_stab) != s:
        raise InsufficientData('Choi stabilizers have a nontrivial output-only element')
    axes, axis_signs = [], []
    for q in range(s, N):
        try:
            coefficients = solve_binary(reference_stab, gz[q, reference])
        except ValueError as exc:
            raise InsufficientData('Cannot move a magic rotation axis to the output') from exc
        multiplier, sign = _product(stab, stab_signs, coefficients)
        label, sign = multiply_paulis(gz[q], int(gz_signs[q]), multiplier, sign)
        if np.any(label[reference]):
            raise InsufficientData('Reference cancellation failed')
        axes.append(label[output])
        axis_signs.append(sign)
    axes = np.asarray(axes, dtype=np.uint8).reshape(t, 2 * n)
    axis_signs = np.asarray(axis_signs, dtype=int)
    if rank(axes) != t or np.any(mm(mm(axes, symplectic_form(n)), axes.T)):
        raise InsufficientData('Output rotation axes are not independent and commuting')

    # Removing the rotations turns the known Choi preparation into the known
    # stabilizer state G |0>^s |+>^t. No physical inverse query is made.
    seed_stab = z_labels.copy()
    for q in range(s, N):
        seed_stab[q, N + q] = 0
        seed_stab[q, q] = 1
    cstar_stab, cstar_signs = conjugate_paulis(seed_stab, np.ones(N, dtype=int), G)
    projection = cstar_stab[:, reference].T
    if rank(projection) != N:
        raise InsufficientData('Learned stabilizer Choi state is not maximally entangled')
    images, image_signs = [], []
    for generator in np.eye(N, dtype=np.uint8):
        coefficients = solve_binary(projection, generator)
        label, sign = _product(cstar_stab, cstar_signs, coefficients)
        # The reference generator is X or Z, hence equals its transpose.
        images.append(label[output])
        image_signs.append(sign)
    Cstar = synthesize_clifford(np.asarray(images, dtype=np.uint8).T,
                               np.asarray(image_signs, dtype=int))
    A = extend_commuting_axes(axes, axis_signs)
    Cin = Cstar + inverse_circuit(A)
    bound = float(delta) if bell_shots >= default_bell and pauli_shots >= default_pauli else None
    return UnitaryRecovery(n, t, Cin, A, Cstar, axes, axis_signs, G, structure,
                           bell_shots, pauli_shots, float(delta), bound)
