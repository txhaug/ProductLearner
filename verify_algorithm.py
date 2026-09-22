#!/usr/bin/env python3
"""Reproducible state-vector verification of Clifford-product cut recovery.

Run: python verify_algorithm.py --trials 3 --samples 65536 --output results.json
The learner receives samples only. Full-support/state-vector operations below
are independent validation oracles, never inputs to recover_cuts.
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
import numpy as np

from clifford_learning import (InsufficientData, circuit_symplectic, column_basis,
    inverse, mm, quadratic_features, rank, recover_cuts, rref,
    symplectic_form, synthesize_decoder, transform_labels)

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.diag([1, 1j])
CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
               [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
SWAP = np.eye(4, dtype=complex)[[0, 2, 1, 3]]
GATES = {'H': H, 'S': S, 'SDG': S.conj().T, 'CX': CX, 'SWAP': SWAP}
ZERO = np.array([1, 0], dtype=complex)
ONE = np.array([0, 1], dtype=complex)
PLUS = H @ ZERO
YPLUS = S @ PLUS
TSTATE = np.array([1, np.exp(1j*np.pi/4)]) / np.sqrt(2)


def apply_gate(state, gate):
    """State axes are q0,q1,..., so q0 is the most significant basis bit."""
    n = len(state).bit_length()-1
    kind, *sites = gate
    order = sites + [i for i in range(n) if i not in sites]
    tensor = state.reshape([2]*n).transpose(order).reshape(2**len(sites), -1)
    return (GATES[kind] @ tensor).reshape([2]*n).transpose(np.argsort(order)).ravel()


def apply_circuit(state, circuit):
    out = state.copy()
    for gate in circuit:
        out = apply_gate(out, gate)
    return out


def inverse_circuit(circuit):
    return [(('SDG' if g[0] == 'S' else 'S' if g[0] == 'SDG' else g[0]), *g[1:])
            for g in reversed(circuit)]


def tensor_product(states):
    out = np.array([1], dtype=complex)
    for state in states:
        out = np.kron(out, state)
    return out


def haar_state(n, rng):
    v = rng.normal(size=2**n) + 1j*rng.normal(size=2**n)
    return v / np.linalg.norm(v)


def random_clifford(n, rng, length=None):
    """Random H/S/CX circuit, NOT a uniform sample from the Clifford group."""
    out = []
    for _ in range(12*n*n if length is None else length):
        gate_type = int(rng.integers(3 if n > 1 else 2))
        if gate_type < 2:
            out.append((('H', 'S')[gate_type], int(rng.integers(n))))
        else:
            a, b = map(int, rng.choice(n, 2, replace=False))
            out.append(('CX', a, b))
    return out


def labels(n):
    """Integer label bit i is x_i; bit n+i is z_i (independent of state indexing)."""
    return ((np.arange(4**n, dtype=np.uint64)[:, None] >>
             np.arange(2*n, dtype=np.uint64)) & 1).astype(np.uint8)


def label_ids(records):
    return records.astype(np.int64) @ (1 << np.arange(records.shape[1], dtype=np.int64))


def bell_distribution(state):
    """Actual 2n-qubit Bell-measurement circuit on |psi> tensor |psi>.

    Apply CX(q,q+n), then H(q); the measured first/second copy bits are z/x.
    No conjugated copy is used. Costs O(n*4^n) time and O(4^n) memory.
    """
    n = len(state).bit_length()-1
    copies = np.kron(state, state)
    for q in range(n):
        copies = apply_gate(copies, ('CX', q, n+q))
        copies = apply_gate(copies, ('H', q))
    lab = labels(n)
    weights = 1 << np.arange(n-1, -1, -1)
    x, z = lab[:, :n] @ weights, lab[:, n:] @ weights
    probabilities = abs(copies[(z << n) | x])**2
    return probabilities / probabilities.sum()


def walsh_transform(v):
    v = np.asarray(v, dtype=complex).copy()
    stride = 1
    while stride < len(v):
        chunks = v.reshape(-1, 2*stride)
        a, b = chunks[:, :stride].copy(), chunks[:, stride:].copy()
        chunks[:, :stride], chunks[:, stride:] = a+b, a-b
        stride *= 2
    return v


def pauli_amplitudes(state, identical_copies):
    """Independent Walsh evaluation of psi^T W psi or psi^dagger W psi.

    Only magnitudes are needed, so the unit-modulus Hermitian-Pauli phase
    i^(x dot z) can be omitted.
    """
    n = len(state).bit_length()-1
    d = 2**n
    amplitudes = np.empty((d, d), dtype=complex)
    bra = state if identical_copies else state.conj()
    for x in range(d):
        amplitudes[x] = walsh_transform(bra[np.arange(d) ^ x] * state)
    lab = labels(n)
    weights = 1 << np.arange(n-1, -1, -1)
    return amplitudes[lab[:, :n] @ weights, lab[:, n:] @ weights]


def support_records(probabilities, n, tolerance):
    return labels(n)[probabilities > tolerance]


def canonical_space(v):
    return (v.shape[0], rank(v), rref(v.T)[0].tobytes())


def all_cut_spaces(spaces, dimension):
    """Exponential enumeration used ONLY to verify small-case answers."""
    out = set()
    for mask in range(1 << len(spaces)):
        selected = [v for i, v in enumerate(spaces) if (mask >> i) & 1]
        v = np.concatenate(selected, axis=1) if selected else np.zeros((dimension, 0), np.uint8)
        out.add(canonical_space(v))
    return out


@dataclass
class Case:
    name: str
    factors: list
    # Each flag marks a single-qubit stabilizer factor in the latent product.
    stabilized: list
    internal_clifford: list

    @property
    def n(self):
        return sum(len(v).bit_length()-1 for v in self.factors)

    @property
    def block_sizes(self):
        return [len(v).bit_length()-1 for v in self.factors]


def make_cases(rng):
    theta = 0.31
    return [
        Case('single_T', [TSTATE], [False], []),
        Case('single_stabilizer', [YPLUS], [True], []),
        Case('T_product_6', [TSTATE]*6, [False]*6, []),
        Case('Haar_qubit_product_6', [haar_state(1, rng) for _ in range(6)], [False]*6, []),
        Case('stabilizer_product_6', [ZERO, ONE, PLUS, YPLUS, ZERO, PLUS], [True]*6, []),
        Case('mixed_stabilizer_and_magic_6', [ZERO, ONE, YPLUS, TSTATE,
             haar_state(1, rng), TSTATE], [True]*3+[False]*3, []),
        Case('Haar_blocks_2_2_2', [haar_state(2, rng) for _ in range(3)], [False]*3, []),
        Case('Haar_blocks_1_2_3', [haar_state(k, rng) for k in (1, 2, 3)], [False]*3, []),
        Case('Haar_blocks_2_3_3', [haar_state(k, rng) for k in (2, 3, 3)], [False]*3, []),
        Case('single_Haar_block_6', [haar_state(6, rng)], [False], []),
        Case('GHZ_Bell_and_magic_7', [ZERO]*5+[TSTATE, haar_state(1, rng)],
             [True]*5+[False]*2, [('H', 0), ('CX', 0, 1), ('CX', 0, 2),
                                 ('H', 3), ('CX', 3, 4)]),
        Case('entangled_width_one_input',
             [np.array([np.cos(theta), np.sin(theta)]), ZERO, TSTATE, haar_state(2, rng)],
             [False, True, False, False], [('CX', 0, 1)]),
    ]


def state_checks(state, recovery, rng):
    decoded = apply_circuit(state, recovery.decoder)
    n, s = recovery.n, recovery.stabilizer_rank
    factors, purity_errors = [], []
    for block in recovery.blocks:
        order = block + [i for i in range(n) if i not in block]
        matrix = decoded.reshape([2]*n).transpose(order).reshape(2**len(block), -1)
        u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
        factors.append(u[:, 0])
        purity_errors.append(max(0., 1-float(np.sum(singular**4))))
    product = tensor_product(factors)
    infidelity = max(0., 1-float(abs(np.vdot(decoded, product))**2))
    # One joint computational-basis measurement supplies all generator signs.
    probabilities = abs(decoded)**2
    outcome = int(rng.choice(len(decoded), p=probabilities/probabilities.sum()))
    signs = [(outcome >> (n-1-i)) & 1 for i in range(s)]
    fixed_mass_error = 0.
    if s:
        selected = outcome >> (n-s)
        fixed_mass_error = max(0., 1-float(probabilities.reshape(2**s, -1)[selected].sum()))
    return {'max_block_impurity': max(purity_errors, default=0.),
            'product_reconstruction_infidelity': infidelity,
            'joint_sign_readout_bits': signs,
            'fixed_qubit_probability_error': fixed_mass_error}


def compare_true_subsystems(case, preparation, recovery):
    """Compare intrinsic residual subspaces modulo the recovered stabilizers.

    Stabilizer removal can add fixed-qubit Z components to logical labels.
    Those components are quotiented out; comparing physical block names would
    incorrectly reject equivalent valid decoders.
    """
    n, s = recovery.n, recovery.stabilizer_rank
    change = mm(circuit_symplectic(n, recovery.decoder), circuit_symplectic(n, preparation))
    keep = list(range(s, n)) + list(range(n+s, 2*n))
    expected = []
    offset = 0
    for size, fixed in zip(case.block_sizes, case.stabilized):
        sites = list(range(offset, offset+size))
        if not fixed:
            expected.append(column_basis(change[np.ix_(keep, sites+[n+i for i in sites])]))
        offset += size
    actual = []
    for block in recovery.blocks[s:]:
        sites = [q-s for q in block]
        actual.append(np.eye(2*(n-s), dtype=np.uint8)[:, sites+[n-s+i for i in sites]])
    atom_match = {canonical_space(v) for v in expected} == {canonical_space(v) for v in actual}
    cuts_match = all_cut_spaces(expected, 2*(n-s)) == all_cut_spaces(actual, 2*(n-s))
    return atom_match, cuts_match, 2**len(expected)


def run_case(case, rng, samples, tolerance):
    started = time.perf_counter()
    n = case.n
    latent = tensor_product(case.factors)
    preparation = case.internal_clifford + random_clifford(n, rng)
    state = apply_circuit(latent, preparation)
    state /= np.linalg.norm(state)
    inverse_error = np.linalg.norm(apply_circuit(state, inverse_circuit(preparation))-latent)
    probs = bell_distribution(state)
    direct = abs(pauli_amplitudes(state, True))**2 / (2**n)
    distribution_error = float(np.max(abs(probs-direct)))
    full_records = support_records(probs, n, tolerance)
    # Verify that the chosen support threshold is not deciding this result.
    threshold_stable = (np.array_equal(probs > tolerance, probs > tolerance*100)
                        and np.array_equal(probs > tolerance, probs > tolerance/100))
    oracle = recover_cuts(full_records)
    sampled_ids = rng.choice(len(probs), size=samples, p=probs)
    records = labels(n)[sampled_ids]
    empirical_features, _ = quadratic_features(np.unique(records, axis=0))
    full_features, _ = quadratic_features(full_records)
    rank_match = rank(empirical_features) == rank(full_features)
    # Independent Pauli-expectation oracle, not the Bell-kernel calculation.
    stabilized_labels = labels(n)[abs(abs(pauli_amplitudes(state, False))**2-1) < 1e-10]
    true_stabilizers = column_basis(stabilized_labels.T)
    base = {'case': case.name, 'n': n, 'samples': samples, 'copies_for_Bell_data': 2*samples,
            'preparation_gates': len(preparation), 'Bell_support_size': len(full_records),
            'smallest_resolved_Bell_probability': float(probs[probs > tolerance].min()),
            'support_threshold_stable': bool(threshold_stable),
            'Bell_circuit_vs_Walsh_max_error': distribution_error,
            'inverse_preparation_vector_error': float(inverse_error),
            'full_quadratic_feature_rank': rank(full_features),
            'sampled_quadratic_feature_rank': rank(empirical_features),
            'empirical_kernel_equals_full_support_kernel': bool(rank_match),
            'oracle_block_sizes': sorted(map(len, oracle.blocks)),
            'expected_block_sizes': sorted(case.block_sizes)}
    oracle_atom, oracle_cuts, _ = compare_true_subsystems(case, preparation, oracle)
    oracle_checks = state_checks(state, oracle, rng)
    oracle_valid = (oracle_atom and oracle_cuts and oracle.stabilizer_rank == sum(case.stabilized)
                    and oracle_checks['product_reconstruction_infidelity'] < 1e-9
                    and canonical_space(oracle.stabilizers.T) == canonical_space(true_stabilizers))
    base['full_support_oracle_passed'] = bool(oracle_valid)
    try:
        recovered = recover_cuts(records)
        checks = state_checks(state, recovered, rng)
        atoms, cuts, count = compare_true_subsystems(case, preparation, recovered)
        stab_match = canonical_space(recovered.stabilizers.T) == canonical_space(true_stabilizers)
        passed = (oracle_valid and distribution_error < 1e-10 and inverse_error < 1e-10
                  and threshold_stable and rank_match and atoms and cuts and stab_match
                  and checks['max_block_impurity'] < 1e-9
                  and checks['product_reconstruction_infidelity'] < 1e-9
                  and checks['fixed_qubit_probability_error'] < 1e-9)
        base.update(checks)
        base.update({'passed': bool(passed), 'stabilizer_rank': recovered.stabilizer_rank,
                     'nullity': recovered.nullity, 'recovered_width': recovered.width,
                     'recovered_block_sizes': sorted(map(len, recovered.blocks)),
                     'stabilizer_space_matches_Pauli_oracle': bool(stab_match),
                     'irreducible_subspaces_match_preparation': bool(atoms),
                     'all_residual_cuts_match_preparation': bool(cuts),
                     'number_of_residual_cuts_checked': count,
                     'decoder_gate_count': len(recovered.decoder),
                     'decoder_gates': recovered.decoder,
                     'decoded_blocks': recovered.blocks})
    except InsufficientData as exc:
        base.update({'passed': False, 'recovery_error': str(exc)})
    base['seconds'] = time.perf_counter()-started
    return base


def regression_checks(rng):
    """Focused checks for conventions and the nontrivial synthesis step."""
    for n in (1, 2, 4):
        j = symplectic_form(n)
        for _ in range(5):
            circuit = random_clifford(n, rng)
            f = circuit_symplectic(n, circuit)
            assert np.array_equal(mm(mm(f.T, j), f), j)
            decoder = synthesize_decoder(f)
            assert np.array_equal(mm(circuit_symplectic(n, decoder), f), np.eye(2*n, dtype=np.uint8))
    psi = haar_state(3, rng)
    original = bell_distribution(psi)
    for gate in [('H', 0), ('S', 1), ('SDG', 2), ('CX', 0, 2), ('SWAP', 1, 2)]:
        transformed = labels(3)
        transform_labels(transformed, gate, affine=True)
        relabeled = np.zeros_like(original)
        relabeled[label_ids(transformed)] = original
        assert np.max(abs(relabeled-bell_distribution(apply_gate(psi, gate)))) < 1e-12
    # Identical-copy Bell sampling must NOT be replaced by Pauli sampling.
    p = bell_distribution(YPLUS)
    wrong = abs(pauli_amplitudes(YPLUS, False))**2 / 2
    assert np.max(abs(p-wrong)) > 0.4
    assert np.max(abs(p-abs(pauli_amplitudes(YPLUS, True))**2/2)) < 1e-12
    try:
        recover_cuts(np.array([[0, 2]], dtype=int))
    except ValueError:
        pass
    else:
        raise AssertionError('Nonbinary input accepted')
    return {'random_symplectic_syntheses_checked': 15,
            'Bell_affine_gate_conventions_checked': 5,
            'identical_copy_vs_conjugate_copy_regression': True}


def sampling_stress(rng, tolerance):
    """Deliberate small-gap case: no assertion of exact learning from few copies."""
    angle = 1e-3
    weak = np.exp(1j*angle*np.array([1, -1, -1, 1])) * tensor_product([TSTATE, TSTATE])
    state = apply_circuit(weak, random_clifford(2, rng))
    probs = bell_distribution(state)
    oracle = recover_cuts(support_records(probs, 2, tolerance))
    records = labels(2)[rng.choice(16, size=512, p=probs)]
    result = {'description': 'Weak ZZ entangling rotation on two T states',
              'angle_radians': angle, 'samples': 512,
              'oracle_width': oracle.width,
              'warning': 'Empirical projector checks can pass for a false exact cut.'}
    try:
        learned = recover_cuts(records)
        checks = state_checks(state, learned, rng)
        result.update(checks)
        result['sampled_width'] = learned.width
        result['exact_recovery_succeeded'] = learned.width == oracle.width
        result['algebra_checks_passed'] = True
    except InsufficientData as exc:
        result.update({'algebra_checks_passed': False, 'recovery_error': str(exc),
                       'exact_recovery_succeeded': False})
    assert oracle.width == 2
    # A second stress case probes the affine gap: almost an additional stabilizer.
    theta = 1e-3
    near_axis = np.array([np.cos(theta), np.exp(0.37j)*np.sin(theta)])
    near_axis = apply_circuit(near_axis, random_clifford(1, rng))
    probabilities = bell_distribution(near_axis)
    reference = recover_cuts(support_records(probabilities, 1, tolerance))
    measured = labels(1)[rng.choice(4, size=512, p=probabilities)]
    affine_result = {'description': 'One qubit close to a Pauli eigenstate',
                     'angle_radians': theta, 'samples': 512,
                     'oracle_stabilizer_rank': reference.stabilizer_rank,
                     'warning': 'Rare affine violations can hide a nonstabilizer component.'}
    try:
        inferred = recover_cuts(measured)
        affine_result.update(state_checks(near_axis, inferred, rng))
        affine_result.update({'sampled_stabilizer_rank': inferred.stabilizer_rank,
                              'exact_recovery_succeeded': inferred.stabilizer_rank == reference.stabilizer_rank,
                              'algebra_checks_passed': True})
    except InsufficientData as exc:
        affine_result.update({'algebra_checks_passed': False, 'recovery_error': str(exc),
                              'exact_recovery_succeeded': False})
    assert reference.stabilizer_rank == 0
    return [result, affine_result]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20260914)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--samples', type=int, default=65536)
    parser.add_argument('--max-qubits', type=int, default=8)
    parser.add_argument('--case', action='append', help='Only run named cases; repeatable')
    parser.add_argument('--support-tolerance', type=float, default=1e-24)
    parser.add_argument('--output', type=Path, default=Path('results.json'))
    args = parser.parse_args()
    if args.trials < 1 or args.samples < 1 or not 1 <= args.max_qubits <= 10:
        parser.error('Require trials,samples >= 1 and 1 <= max-qubits <= 10')
    if not 0 < args.support_tolerance < 1e-12:
        parser.error('Support tolerance must lie between 0 and 1e-12')
    master = np.random.SeedSequence(args.seed)
    streams = master.spawn(args.trials+2)
    regression = regression_checks(np.random.default_rng(streams[0]))
    results = []
    for trial in range(args.trials):
        rng = np.random.default_rng(streams[trial+1])
        cases = make_cases(rng)
        if args.case and not set(args.case).issubset({c.name for c in cases}):
            parser.error('Unknown case name')
        for case in cases:
            if case.n > args.max_qubits or (args.case and case.name not in args.case):
                continue
            row = run_case(case, rng, args.samples, args.support_tolerance)
            row['trial'] = trial
            results.append(row)
            print(f"{'PASS' if row['passed'] else 'FAIL'} trial={trial} {case.name}: "
                  f"blocks={row.get('recovered_block_sizes')} "
                  f"rank={row['sampled_quadratic_feature_rank']}/{row['full_quadratic_feature_rank']}", flush=True)
    if not results:
        parser.error('No cases selected')
    stress = sampling_stress(np.random.default_rng(streams[-1]), args.support_tolerance)
    passed = sum(r['passed'] for r in results)
    report = {'configuration': vars(args) | {'output': str(args.output)},
              'access_model': 'recover_cuts receives Bell records only; state-vector checks are validation oracles',
              'Clifford_ensemble': 'random H/S/CX circuits, not uniform group samples',
              'support_oracle': 'floating-point probabilities above stated threshold, with 100x threshold checks',
              'regressions': regression,
              'summary': {'passed': passed, 'total': len(results),
                          'all_passed': passed == len(results),
                          'max_Bell_probability_error': max(r['Bell_circuit_vs_Walsh_max_error'] for r in results),
                          'max_product_infidelity': max((r.get('product_reconstruction_infidelity', 0.) for r in results), default=0.)},
              'cases': results, 'small_gap_stress': stress}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['summary'], indent=2))
    print('Small-gap stress:', json.dumps(stress, indent=2))
    raise SystemExit(0 if passed == len(results) else 1)


if __name__ == '__main__':
    main()
