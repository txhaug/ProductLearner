#!/usr/bin/env python3
"""Verify T-depth-one unitary learning using simulated measurements.

Run ``python verify_unitary_learning.py --max-qubits 4 --trials 2``.

The learner receives only sampled Bell and Pauli measurement outcomes.
It cannot access the target circuit, unitary matrix, or state vector.
These are used by the simulator to generate measurements and independently
verify the reconstructed circuit.

Each test also reruns the learner using the recorded measurement outcomes
and checks that it returns the same circuit.

The state-vector simulator has exponential cost, while the learning
algorithm uses polynomial classical processing and polynomially many
queries to the unknown unitary. Failed trials are reported without retries.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np

from clifford_learning import InsufficientData
from unitary_learning import learn_t_depth_one
from unitary_simulator import StatevectorUnitaryOracle, apply_circuit, circuit_matrix
from verify_algorithm import random_clifford


class RecordingMeasurements:
    """Provide n and the two measurement operations to the learner.

    The transcript records experimental settings and outcomes, not target
    amplitudes or gates. This is an interface boundary, not a security sandbox.
    """
    __slots__ = ('n', '_backend', 'transcript')

    def __init__(self, backend):
        self.n = backend.n
        self._backend = backend
        self.transcript = []

    def sample_bell(self, shots, rng):
        result = self._backend.sample_bell(shots, rng)
        self.transcript.append(('bell', int(shots), result.copy()))
        return result

    def sample_decoded_paulis(self, decoder, bases, shots, rng):
        result = self._backend.sample_decoded_paulis(decoder, bases, shots, rng)
        settings = (tuple(tuple(g) for g in decoder), tuple(bases), int(shots))
        self.transcript.append(('pauli', settings, result.copy()))
        return result


class ReplayMeasurements:
    """An oracle containing only previously sampled outcomes and their settings."""
    __slots__ = ('n', 'transcript', 'position')

    def __init__(self, n, transcript):
        self.n = n
        self.transcript = transcript
        self.position = 0

    def _next(self, kind, settings):
        if self.position >= len(self.transcript):
            raise AssertionError('Learner requested more measurements during replay')
        expected_kind, expected_settings, outcomes = self.transcript[self.position]
        if kind != expected_kind or settings != expected_settings:
            raise AssertionError('Replay measurement settings changed')
        self.position += 1
        return outcomes.copy()

    def sample_bell(self, shots, rng):
        return self._next('bell', int(shots))

    def sample_decoded_paulis(self, decoder, bases, shots, rng):
        settings = (tuple(tuple(g) for g in decoder), tuple(bases), int(shots))
        return self._next('pauli', settings)


def _random_state(dimension, rng):
    state = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
    return state / np.linalg.norm(state)


def _phase_and_error(target, candidate):
    overlap = np.vdot(target, candidate)
    phase = overlap.conjugate() / abs(overlap) if abs(overlap) else 1.0
    return phase, float(np.linalg.norm(target - phase * candidate) / np.linalg.norm(target))


def _pauli_matrix(label):
    """Independent dense Hermitian-Pauli construction for verification only."""
    matrices = {(0, 0): np.eye(2), (1, 0): np.array([[0, 1], [1, 0]]),
                (0, 1): np.diag([1, -1]),
                (1, 1): np.array([[0, -1j], [1j, 0]])}
    n = len(label) // 2
    result = np.ones((1, 1), dtype=complex)
    for q in range(n):
        result = np.kron(result, matrices[(int(label[q]), int(label[n + q]))])
    return result


def validation_metrics(target, recovered, rng, random_inputs=4):
    """Compare complete channels and arbitrary inputs, never feeding truth back."""
    n, dimension = recovered.n, 2 ** recovered.n
    target = np.asarray(target)
    if target.shape != (dimension, dimension):
        raise ValueError('Target matrix does not match the recovered number of qubits')
    candidate = circuit_matrix(n, recovered.circuit)
    phase, matrix_error = _phase_and_error(target, candidate)
    overlap = np.vdot(target, candidate) / dimension
    input_errors = []
    for _ in range(random_inputs):
        state = _random_state(dimension, rng)
        input_errors.append(float(np.linalg.norm(target @ state - phase * candidate @ state)))
    # An untouched external reference tests the coherent action, not just output
    # probabilities or the learned unitary on a single preparation input.
    entangled = _random_state(2 * dimension, rng)
    entangled_error = float(np.linalg.norm(
        np.kron(np.eye(2), target) @ entangled
        - phase * np.kron(np.eye(2), candidate) @ entangled))

    s = recovered.structure.stabilizer_rank
    seed = np.ones(1, dtype=complex)
    for q in range(2 * n):
        qubit = (np.array([1, 0], dtype=complex) if q < s
                 else np.array([1, np.exp(1j * np.pi / 4)]) / np.sqrt(2))
        seed = np.kron(seed, qubit)
    learned_choi = apply_circuit(seed, recovered.choi_preparation)
    true_choi = target.T.ravel() / np.sqrt(dimension)
    _, choi_preparation_error = _phase_and_error(true_choi, learned_choi)

    rotations = np.eye(dimension, dtype=complex)
    for label, sign in zip(recovered.rotation_axes, recovered.rotation_signs):
        rotation = (np.cos(np.pi / 8) * np.eye(dimension)
                    - 1j * np.sin(np.pi / 8) * int(sign) * _pauli_matrix(label))
        rotations = rotation @ rotations
    factorization = rotations @ circuit_matrix(n, recovered.remaining_clifford)
    _, axis_error = _phase_and_error(target, factorization)
    return {
        'unitary_relative_frobenius_error_up_to_phase': matrix_error,
        'choi_infidelity': float(max(0.0, 1 - abs(overlap) ** 2)),
        'maximum_random_input_error': max(input_errors, default=0.0),
        'external_reference_input_error': entangled_error,
        'learned_choi_preparation_error': choi_preparation_error,
        'signed_rotation_factorization_error': axis_error,
    }


@dataclass
class Case:
    name: str
    n: int
    t: int
    gates: list
    t_sites: list


def deterministic_cases(max_qubits):
    cases = [Case('identity_1', 1, 0, [], []),
             Case('T_1', 1, 1, [('T', 0)], [0]),
             Case('Tdagger_1', 1, 1, [('TDG', 0)], [0]),
             Case('phase_sensitive_Clifford_1', 1, 0,
                  [('H', 0), ('S', 0), ('Y', 0), ('SDG', 0)], []),
             Case('phase_sensitive_T_1', 1, 1,
                  [('S', 0), ('H', 0), ('T', 0), ('Y', 0), ('SDG', 0), ('H', 0)], [0])]
    if max_qubits >= 2:
        cases.extend([
            Case('identity_2', 2, 0, [], []),
            Case('nonprefix_T_mask_2', 2, 1,
                 [('H', 0), ('CX', 0, 1), ('S', 1), ('T', 1),
                  ('CX', 1, 0), ('H', 1), ('Y', 0)], [1]),
            Case('mixed_T_and_Tdagger_2', 2, 2,
                 [('H', 0), ('S', 1), ('CX', 1, 0), ('T', 0), ('TDG', 1),
                  ('CX', 0, 1), ('S', 0), ('H', 1)], [0, 1]),
        ])
    return cases


def random_cases(max_qubits, trials, rng):
    for n in range(1, max_qubits + 1):
        for t in range(n + 1):
            for trial in range(trials):
                sites = sorted(map(int, rng.choice(n, size=t, replace=False)))
                cin = random_clifford(n, rng)
                cout = random_clifford(n, rng)
                gates = cin + [('T', q) for q in sites] + cout
                yield Case(f'random_n{n}_t{t}_trial{trial}', n, t, gates, sites)


def run_case(case, rng, *, delta=.01, bell_shots=None, pauli_shots=None,
             tolerance=1e-9):
    started = time.perf_counter()
    backend = StatevectorUnitaryOracle.from_gates(case.n, case.gates)
    measurements = RecordingMeasurements(backend)
    result = {'case': case.name, 'n': case.n, 'target_t': case.t,
              'target_t_sites': case.t_sites}
    options = dict(delta=delta, bell_shots=bell_shots, pauli_shots=pauli_shots)
    try:
        recovered = learn_t_depth_one(measurements, rng=rng, **options)
    except InsufficientData as exc:
        result.update(status='insufficient_data', passed=False, reason=str(exc),
                      query_counts=dict(backend.counts),
                      seconds=time.perf_counter() - started)
        return result
    target = circuit_matrix(case.n, case.gates)
    metrics = validation_metrics(target, recovered, rng)
    replay = ReplayMeasurements(case.n, measurements.transcript)
    replayed = learn_t_depth_one(replay, rng=0, **options)
    replay_ok = replay.position == len(measurements.transcript) and replayed.circuit == recovered.circuit
    expected = {'bell_records': recovered.bell_records,
                'decoded_pauli_shots': 3 * recovered.pauli_shots_per_basis,
                'forward_queries': recovered.forward_queries}
    counts_ok = backend.counts == expected
    passed = (recovered.t == case.t and replay_ok and counts_ok
              and all(value < tolerance for value in metrics.values()))
    result.update(status='success' if passed else 'verification_failed', passed=bool(passed),
                  metrics=metrics, transcript_replay_identical=bool(replay_ok),
                  query_counts=dict(backend.counts), query_counts_correct=bool(counts_ok),
                  simulator_forward_evaluations=backend.simulator_forward_evaluations,
                  reconstruction=recovered.to_dict(), seconds=time.perf_counter() - started)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-qubits', type=int, default=4,
                        help='largest target unitary; Bell simulation uses 4n qubits (default: 4)')
    parser.add_argument('--trials', type=int, default=1, help='random trials for each (n,t)')
    parser.add_argument('--seed', type=int, default=20260922)
    parser.add_argument('--delta', type=float, default=.01, help='failure budget per case')
    parser.add_argument('--bell-shots', type=int, help='override default; can remove guarantee')
    parser.add_argument('--pauli-shots', type=int, help='override shots PER Pauli setting')
    parser.add_argument('--output', type=Path, default=Path('unitary_results.json'))
    args = parser.parse_args()
    if not 1 <= args.max_qubits <= 5 or args.trials < 1:
        parser.error('Require 1 <= max-qubits <= 5 and trials >= 1')
    if not 0 < args.delta < 1:
        parser.error('Require 0 < delta < 1')
    if any(value is not None and value < 1 for value in (args.bell_shots, args.pauli_shots)):
        parser.error('Shot overrides must be positive')
    streams = np.random.SeedSequence(args.seed).spawn(2)
    cases = deterministic_cases(args.max_qubits) + list(random_cases(
        args.max_qubits, args.trials, np.random.default_rng(streams[0])))
    case_streams = streams[1].spawn(len(cases))
    results = []
    for case, stream in zip(cases, case_streams):
        result = run_case(case, np.random.default_rng(stream), delta=args.delta,
                          bell_shots=args.bell_shots, pauli_shots=args.pauli_shots)
        results.append(result)
        print(f"{case.name}: {result['status']}, queries={result['query_counts']['forward_queries']}",
              flush=True)
    output = {
        'configuration': vars(args) | {'output': str(args.output)},
        'seed': args.seed, 'delta_per_case': args.delta,
        'scope': 'Exact-promise finite-sample learner, exponential state-vector validation',
        'clifford_ensemble': 'Random H/S/CX circuits; not uniform Clifford group samples',
        'retry_policy': 'No retries or adaptive shot increases',
        'simulation_counting': ('Forward queries count the represented experiment; cached '
                                'state-vector preparations count separately.'),
        'summary': {'total_cases': len(results),
                    'passed': sum(result['passed'] for result in results),
                    'all_passed': all(result['passed'] for result in results)},
        'results': results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(f"Saved {args.output}; {output['summary']['passed']}/{len(results)} passed.")
    return 0 if output['summary']['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
