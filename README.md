# Learning Clifford-scrambled states and T-depth-one unitaries

This code accompanies the paper "Efficient Learning of Clifford-Scrambled Product States"

Author: Tobias Haug, Technology Innovation Institute, tobias.haug@u.nus.edu

This code was written with assistance from ChatGPT 5.6.

Python implementations of the learning algorithms in *Efficient Learning of
Clifford-Scrambled Product States*, with state-vector simulations and independent
correctness checks.

The package supports two tasks:

- **State structure recovery:** infer Pauli stabilizers, hidden product blocks,
  and a Clifford decoder from two-copy Bell measurement outcomes.
- **Unitary learning:** reconstruct an unknown circuit
  `U = C_out @ T_layer @ C_in`, including both Clifford layers and the number
  of T gates, using forward queries and sampled Bell and Pauli measurements.

The state learner returns the hidden subsystem structure. The unitary learner
returns an explicit circuit that implements the target unitary up to global
phase. Both use binary linear algebra and Clifford synthesis.

## Installation

Python 3.10+ and NumPy are required. 

## Quick start: unitary learning

Gate lists specify operations in application order: input Clifford, parallel
T layer, then output Clifford.

```python
from unitary_learning import learn_t_depth_one
from unitary_simulator import StatevectorUnitaryOracle

# The target circuit is supplied only to the measurement simulator.
target = [('H', 0), ('CX', 0, 1),
          ('T', 1),
          ('S', 0), ('CX', 1, 0), ('H', 1)]
oracle = StatevectorUnitaryOracle.from_gates(2, target)
answer = learn_t_depth_one(oracle, delta=0.01, rng=123)

print('T count:', answer.t)
print('Input Clifford:', answer.input_clifford)
print('Output Clifford:', answer.output_clifford)
print('Complete circuit:', answer.circuit)
print('Forward queries:', answer.forward_queries)
```

`answer.circuit` applies the returned input Clifford, T gates on qubits
`0,...,t-1`, and the returned output Clifford. The target T gates may be on any
subset of qubits. The reconstructed Clifford layers need not match the target
layers individually; their combined action matches the target up to global
phase. `answer.to_dict()` exports the result as JSON-compatible data.

The input promise is an exact, ancilla-free T-depth-one unitary. The learner
requires no inverse or controlled query to the unknown unitary. It identifies
the finite set of possible decoded magic states using additional Pauli
measurements, including the stabilizer signs that Bell data alone do not give.



## Quick start: state structure recovery

`recover_cuts(records)` accepts a binary array of Bell outcomes. It receives
neither the state vector nor its preparation circuit. This example generates
the records with the simulator:

```python
import numpy as np
from clifford_learning import recover_cuts
from verify_algorithm import (
    TSTATE, apply_circuit, bell_distribution, labels,
    random_clifford, tensor_product,
)

rng = np.random.default_rng(123)
n = 3
psi = apply_circuit(tensor_product([TSTATE] * n), random_clifford(n, rng))
probabilities = bell_distribution(psi)
outcomes = rng.choice(len(probabilities), size=4096, p=probabilities)
records = labels(n)[outcomes]

answer = recover_cuts(records)
print('Stabilizer rank:', answer.stabilizer_rank)
print('Nullity:', answer.nullity)
print('Block sizes:', [len(block) for block in answer.blocks])
print('Decoder:', answer.decoder)
```

`answer.decoder` is a gate list implementing `D†`, with
`D† |psi> = tensor_product(block states)`. Blocks are lists of decoded,
zero-indexed qubit positions; stabilized qubits appear first. The decomposition
is recovered without having to reproduce the preparation circuit gate by gate.

This routine recovers structure, not continuous block-state amplitudes or
stabilizer signs. The state-verification script measures the fixed-qubit signs
and checks the decoded product structure separately. Its extraction of block
vectors from simulated amplitudes is a correctness check, not a finite-copy
block-tomography algorithm.

## Run the verification scripts

Verify unitary learning on deterministic examples and random Clifford layers:

```bash
python verify_unitary_learning.py --max-qubits 4 --trials 2 --output unitary_results.json
```

Verify state structure recovery across twelve input families:

```bash
python verify_algorithm.py --trials 3 --samples 65536 --output results.json
```



Failed verification cases are reported without retries,
and either script exits with code 1 if a main case fails. The state script also
reports separate small-gap stress cases that deliberately demonstrate failures
of exact recovery at insufficient sample sizes.

## What the learner can access

The learner receives sampled measurement outcomes. It cannot access the target
circuit, unitary matrix, or state vector. The simulator uses these to generate
measurements and independently verify the result.

Each unitary test also runs the learner again using the saved outcomes and
measurement settings, checking that it returns the same circuit. Full matrices,
exact-support calculations, and state-vector comparisons are used only for
simulation and verification.

The learning algorithms use polynomial classical processing in the number of
qubits and records. The state-vector simulator is exponential: Bell sampling
uses two copies of an n-qubit state, or two copies of a 2n-qubit Choi state for
an n-qubit unitary. The latter is a **4n-qubit simulation**, with a default limit
of five target qubits. This simulation cost is distinct from the polynomial
query complexity of the quantum learning protocol.

The unitary backend caches simulated states and probabilities. Its experimental
query count still charges two forward queries per Bell record and one per
fresh Choi copy measured in a Pauli setting. Actual simulator callback
invocations are counted separately.

## Algorithms

### State structure recovery

1. Find the nullspace of the affine feature matrix `[1,v]` and convert its
   equations into unsigned Pauli stabilizer labels.
2. Check their commutation and affine constants, synthesize a stabilizer
   decoder, and transform Bell labels using its affine action.
3. Remove the stabilized coordinates and find the nullspace of the residual
   quadratic feature matrix with columns `1`, `v_i`, and `v_i v_j` for `i<j`.
4. Polarize the quadratic equations and check that the resulting maps form the
   required commuting projector algebra. Reject excess empirical equations
   before constructing these maps.
5. Find the common eigenspaces, construct symplectic bases for the recovered
   blocks, and synthesize the Clifford decoder.

Duplicate records are removed before binary elimination because repetitions
leave the equation kernel unchanged. GF(2) calculations use exact binary
arithmetic. No search over n-qubit Clifford circuits or product partitions is
performed.

### T-depth-one unitary learning

1. Apply state structure recovery to Bell samples from normalized Choi copies.
2. Measure decoded Pauli observables to determine stabilizer signs and identify
   each magic qubit among the twelve Clifford images of the T state.
3. Recover signed rotation axes and multiply them by stabilizers to remove
   their reference-register support.
4. Extract the signed Clifford tableau of the remaining stabilizer Choi state.
5. Synthesize the input and output Clifford circuits around a parallel T layer.

The single-qubit identification uses a constant-size Clifford orbit. All
subsequent reconstruction uses polynomial-size binary and signed-Pauli data.

## Conventions

Qubit zero is the most significant computational-basis bit. Circuits are lists
of gate tuples in application order, such as `('H', 0)` or `('CX', 0, 1)`.

An n-qubit Pauli or Bell label is ordered as
`(x_0,...,x_(n-1),z_0,...,z_(n-1))`. The Hermitian Pauli convention is
`W_(x,z) = tensor_i i^(x_i*z_i) X^x_i Z^z_i`, so `(1,1)` denotes Y.
Integer label identifiers store `x_i` in bit i and `z_i` in bit n+i; this is
separate from computational-basis indexing.

Bell sampling uses **two identical copies**, with probabilities
`|psi^T W_v psi|^2 / 2^n`. The circuit applies `CX(i,n+i)` followed by `H(i)`;
the first-copy readout gives z and the second gives x. It does not use a
complex-conjugate copy. For S and S-dagger, Bell labels transform as
`z_i <- z_i + x_i + 1`, whereas Pauli labels transform without the constant 1.

For unitary learning, Choi qubits are ordered reference first, then output:
`R0,...,R(n-1),A0,...,A(n-1)`. Each Bell record therefore contains 4n bits.

Random Clifford circuits in the verification scripts are generated from H, S,
and CX gates with length `12*n*n`. They are not uniform samples from the
Clifford group.

## Included validation results

| Report | Coverage | Result |
|---|---|---|
| `unitary_results.json` | 36 unitary cases, 1–4 target qubits | 36/36 passed; maximum phase-aligned relative Frobenius error below `1.1e-14` |
| `results.json` | 12 state families, three trials, up to eight qubits | 36/36 passed; maximum decoded-product infidelity `4.22e-15` |
| `state_regression.json` | One trial for each of the 12 state families | 12/12 passed |
| `tests.log` | Signed Pauli algebra, synthesis, measurements, and reconstruction | 21 tests passed |

State families include T-state products, Haar-random qubit products, stabilizer
products, mixtures of stabilizer and magic states, Haar-random blocks, and
Clifford-prepared entangled states. Checks compare recovered stabilizers and
cuts with independent references and verify the purity of decoded blocks.

Unitary checks compare the full reconstructed matrix up to global phase,
arbitrary input states, an input entangled with an external reference, the
learned Choi preparation, and the signed rotation factorization. Edge cases
include t=0, T-dagger gates, and T gates on nonconsecutive sites.

These are small-system numerical checks. Simulated amplitudes use floating-point
arithmetic; the binary reconstruction is exact. The state support reference
uses threshold `1e-24` and checks stability under a factor-of-100 change in
both directions. Full-support data are never supplied to the learner.

## Scope and failure handling

State structure recovery has exact finite-sample guarantees under the
manuscript's support-gap assumptions. It may raise `InsufficientData` if the
empirical equations fail the algebraic checks. Passing those checks alone does
not certify correctness: rare outcomes can hide a weakly entangled block or
an almost-stabilized qubit. The supplied small-gap stress cases demonstrate
both effects.

Unitary learning assumes the exact T-depth-one promise. Its default sample
budgets give success probability at least `1-delta` under ideal independent
measurements. Smaller explicit budgets can be used experimentally but remove
that claimed bound. The method is not a general membership certificate for
arbitrary or noisy processes.

The package does not implement the manuscript's general approximate-learning
algorithm, finite-copy tomography of arbitrary block states, or a full LOCC
learner. The unitary implementation may apply a known global Clifford decoder
to fresh Choi copies.

