"""Bell-data-only implementation of exact Clifford-product structure recovery.

Requires Python >=3.10 and NumPy. No state vectors or preparation circuits are
passed to recover_cuts. Binary labels are (x_0,...,x_{n-1},z_0,...,z_{n-1}).
The returned gate list applies the decoder D^dagger, not the preparation D.
Finite data can produce spurious equations: passing algebra checks alone is
not a certificate that every inferred cut is exact.
"""
from dataclasses import dataclass
from itertools import combinations
import numpy as np


class InsufficientData(ValueError):
    """Empirical equations do not have the required exact-support structure."""


def mm(a, b):
    return (np.asarray(a, dtype=np.uint8) @ np.asarray(b, dtype=np.uint8)) & 1


def rref(a):
    """GF(2) row reduction using arbitrary-precision packed Python integers."""
    a = np.asarray(a, dtype=np.uint8)
    if a.ndim != 2:
        raise ValueError('Expected a matrix')
    width = a.shape[1]
    basis = [0] * width
    for packed in np.packbits(a, axis=1, bitorder='little'):
        row = int.from_bytes(packed.tobytes(), 'little')
        while row:
            pivot = (row & -row).bit_length() - 1
            if basis[pivot]:
                row ^= basis[pivot]
            else:
                basis[pivot] = row
                break
    pivots = [i for i, row in enumerate(basis) if row]
    for p in reversed(pivots):
        for q in pivots:
            if q >= p:
                break
            if (basis[q] >> p) & 1:
                basis[q] ^= basis[p]
    rows = np.array([[(basis[p] >> j) & 1 for j in range(width)]
                     for p in pivots], dtype=np.uint8).reshape(len(pivots), width)
    return rows, pivots


def nullspace(a):
    """Columns form a basis of ker(a)."""
    rows, pivots = rref(a)
    free = [j for j in range(a.shape[1]) if j not in pivots]
    out = np.zeros((a.shape[1], len(free)), dtype=np.uint8)
    for k, j in enumerate(free):
        out[j, k] = 1
        for i, p in enumerate(pivots):
            out[p, k] = rows[i, j]
    return out


def column_basis(a):
    return rref(np.asarray(a, dtype=np.uint8).T)[0].T


def rank(a):
    return len(rref(a)[1])


def inverse(a):
    d = a.shape[0]
    rows, pivots = rref(np.concatenate([a, np.eye(d, dtype=np.uint8)], axis=1))
    if pivots[:d] != list(range(d)):
        raise ValueError('Singular binary matrix')
    return rows[:d, d:]


def symplectic_form(n):
    j = np.zeros((2*n, 2*n), dtype=np.uint8)
    j[:n, n:] = np.eye(n, dtype=np.uint8)
    j[n:, :n] = np.eye(n, dtype=np.uint8)
    return j


def transform_labels(labels, gate, affine=False):
    """Conjugate row labels by a Clifford; mutate in place.

    affine=True transforms identical-copy Bell outcomes. In particular S acts
    as z_i <- z_i+x_i+1 on Bell outcomes, but z_i <- z_i+x_i on Pauli labels.
    """
    n = labels.shape[1] // 2
    kind, *sites = gate
    a = sites[0]
    if kind == 'H':
        labels[:, [a, n+a]] = labels[:, [n+a, a]]
    elif kind in ('S', 'SDG'):
        labels[:, n+a] ^= labels[:, a]
        if affine:
            labels[:, n+a] ^= 1
    elif kind == 'CX':
        b = sites[1]
        labels[:, b] ^= labels[:, a]
        labels[:, n+a] ^= labels[:, n+b]
    elif kind == 'SWAP':
        b = sites[1]
        labels[:, [a, b]] = labels[:, [b, a]]
        labels[:, [n+a, n+b]] = labels[:, [n+b, n+a]]
    else:
        raise ValueError(f'Unknown gate {gate}')


def circuit_symplectic(n, gates):
    rows = np.eye(2*n, dtype=np.uint8)
    for gate in gates:
        transform_labels(rows, gate)
    return rows.T


def stabilizer_decoder(stabilizers):
    """Map independent commuting unsigned generators to Z_0,...,Z_{s-1}."""
    rows = stabilizers.copy()
    s, twice_n = rows.shape
    n = twice_n // 2
    gates = []
    def emit(g):
        gates.append(g)
        transform_labels(rows, g)
    for i in range(s):
        for j in range(i):
            if rows[i, n+j]:
                rows[i] ^= rows[j]
        for q in range(i, n):
            if rows[i, q]:
                if rows[i, n+q]:
                    emit(('S', q))
                emit(('H', q))
        support = np.flatnonzero(rows[i, n+i:]) + i
        if not len(support):
            raise InsufficientData('Dependent or invalid stabilizer generators')
        if support[0] != i:
            emit(('SWAP', i, int(support[0])))
        for q in range(i+1, n):
            if rows[i, n+q]:
                emit(('CX', q, i))
    target = np.zeros_like(rows)
    for i in range(s):
        target[i, n+i] = 1
    if not np.array_equal(rows, target):
        raise InsufficientData('Failed to isolate stabilizers')
    return gates


def quadratic_features(records):
    d = records.shape[1]
    pairs = list(combinations(range(d), 2))
    columns = [np.ones(len(records), dtype=np.uint8)]
    columns.extend(records[:, i] for i in range(d))
    columns.extend(records[:, i] & records[:, j] for i, j in pairs)
    return np.column_stack(columns), pairs


def polarization_maps(kernel, pairs, n):
    j = symplectic_form(n)
    out = []
    for c in kernel.T:
        b = np.zeros((2*n, 2*n), dtype=np.uint8)
        for coefficient, (u, v) in zip(c[1+2*n:], pairs):
            b[u, v] = b[v, u] = coefficient
        out.append(mm(j, b))
    return out


def joint_spaces(maps, d):
    eye = np.eye(d, dtype=np.uint8)
    spaces = [eye]
    for e in maps:
        next_spaces = []
        for v in spaces:
            for bit in (0, 1):
                sub = mm(v, nullspace(mm(e ^ (eye * bit), v)))
                if sub.shape[1]:
                    next_spaces.append(sub)
        spaces = next_spaces
    return spaces


def symplectic_basis(space, j):
    """Symplectic Gram--Schmidt in one nondegenerate column space."""
    remaining = column_basis(space)
    xs, zs = [], []
    while remaining.shape[1]:
        a = remaining[:, 0].copy()
        pairings = mm(a @ j, remaining)
        choices = np.flatnonzero(pairings)
        if not len(choices):
            raise InsufficientData('Degenerate candidate subsystem')
        b = remaining[:, choices[0]].copy()
        xs.append(a)
        zs.append(b)
        remaining ^= np.outer(a, mm(b @ j, remaining))
        remaining ^= np.outer(b, mm(a @ j, remaining))
        remaining = column_basis(remaining)
    return xs, zs


def synthesize_decoder(frame):
    """Synthesize H/S/CX/SWAP gates mapping the columns of frame to standard X/Z.

    Phases of the Pauli images are unconstrained; they amount to local Paulis
    in the decoded frame and cannot change product cuts.
    """
    rows = frame.T.copy()
    n = len(rows) // 2
    gates = []
    def emit(g):
        gates.append(g)
        transform_labels(rows, g)
    for i in range(n):
        # Convert X_i image to a single X on site i.
        for q in range(i, n):
            if rows[i, n+q]:
                emit(('S', q) if rows[i, q] else ('H', q))
        support = np.flatnonzero(rows[i, i:n]) + i
        if not len(support):
            raise InsufficientData('Invalid symplectic frame')
        if support[0] != i:
            emit(('SWAP', i, int(support[0])))
        for q in range(i+1, n):
            if rows[i, q]:
                emit(('CX', i, q))
        # Convert its symplectic partner to Z_i while preserving X_i.
        for q in range(i+1, n):
            if rows[n+i, q]:
                if rows[n+i, n+q]:
                    emit(('S', q))
                emit(('H', q))
        for q in range(i+1, n):
            if rows[n+i, n+q]:
                emit(('CX', q, i))
        if rows[n+i, i]:
            emit(('H', i))
            emit(('S', i))
            emit(('H', i))
    if not np.array_equal(rows, np.eye(2*n, dtype=np.uint8)):
        raise InsufficientData('Clifford synthesis failed')
    return gates


@dataclass
class Recovery:
    n: int
    stabilizer_rank: int
    stabilizers: np.ndarray
    decoder: list
    blocks: list
    residual_kernel: np.ndarray
    residual_records: np.ndarray
    projectors: list
    affine_rank: int
    quadratic_rank: int

    @property
    def nullity(self):
        return self.n - self.stabilizer_rank

    @property
    def width(self):
        return max(map(len, self.blocks), default=1)


def recover_cuts(records):
    """Recover cuts from Bell records only. Does not learn continuous block states.

    records: binary M-by-2n array with x coordinates followed by z coordinates.
    Independent generator signs require a separate joint measurement on a fresh
    decoded copy. The simulation harness implements that readout and checks it.
    """
    raw = np.asarray(records)
    if raw.ndim != 2 or len(raw) == 0 or raw.shape[1] == 0 or raw.shape[1] % 2:
        raise ValueError('Expected nonempty M-by-2n records, n >= 1')
    if not np.all((raw == 0) | (raw == 1)):
        raise ValueError('Records must be binary')
    records = np.unique(raw.astype(np.uint8), axis=0)
    n = records.shape[1] // 2
    affine = np.column_stack([np.ones(len(records), dtype=np.uint8), records])
    affine_kernel = nullspace(affine)
    linear = affine_kernel[1:].T
    stabilizers = np.concatenate([linear[:, n:], linear[:, :n]], axis=1)
    s = len(stabilizers)
    j = symplectic_form(n)
    if s > n or np.any(mm(mm(stabilizers, j), stabilizers.T)):
        raise InsufficientData('Empirical affine constraints do not commute')
    q = np.sum(stabilizers[:, :n] & stabilizers[:, n:], axis=1) % 2
    if not np.array_equal(q, affine_kernel[0]):
        raise InsufficientData('Empirical affine constants are not stabilizer constants')
    gates = stabilizer_decoder(stabilizers)
    decoded_records = records.copy()
    for gate in gates:
        transform_labels(decoded_records, gate, affine=True)
    if np.any(decoded_records[:, :s]):
        raise InsufficientData('Stabilized Bell coordinates are not fixed')
    nu = n-s
    keep = list(range(s, n)) + list(range(n+s, 2*n))
    residual = np.unique(decoded_records[:, keep], axis=0)
    blocks = [[i] for i in range(s)]
    if not nu:
        return Recovery(n, s, stabilizers, gates, blocks,
                        np.zeros((1, 0), dtype=np.uint8), residual, [],
                        rank(affine), 1)
    features, pairs = quadratic_features(residual)
    kernel = nullspace(features)
    if kernel.shape[1] > nu:
        raise InsufficientData('Too many empirical quadratic equations; collect more records')
    maps = polarization_maps(kernel, pairs, nu)
    eye = np.eye(2*nu, dtype=np.uint8)
    jr = symplectic_form(nu)
    for e in maps:
        if not np.array_equal(mm(e, e), e):
            raise InsufficientData('Non-idempotent empirical polarization')
        if not np.array_equal(mm(e.T, jr), mm(jr, e)):
            raise InsufficientData('Polarization not symplectically self-adjoint')
    for e, f in combinations(maps, 2):
        if not np.array_equal(mm(e, f), mm(f, e)):
            raise InsufficientData('Empirical projectors do not commute')
    if not maps or rank(np.column_stack([e.ravel() for e in maps])) != len(maps):
        raise InsufficientData('Polarization is not injective')
    span = np.column_stack([e.ravel() for e in maps])
    if rank(np.column_stack([span, eye.ravel()])) != rank(span):
        raise InsufficientData('Full-swap projector missing')
    spaces = joint_spaces(maps, 2*nu)
    if len(spaces) != len(maps):
        raise InsufficientData('Empirical polarization space is not the cut algebra')
    xs, zs, sizes = [], [], []
    for v in spaces:
        if v.shape[1] % 2:
            raise InsufficientData('Odd-dimensional candidate subsystem')
        a, b = symplectic_basis(v, jr)
        xs.extend(a)
        zs.extend(b)
        sizes.append(len(a))
    frame = np.column_stack(xs + zs)
    if not np.array_equal(mm(mm(frame.T, jr), frame), jr):
        raise InsufficientData('Joint subsystem basis is not symplectic')
    inv_frame = inverse(frame)
    projectors = []
    start = 0
    for size in sizes:
        sites = list(range(start, start+size))
        select = np.zeros(2*nu, dtype=np.uint8)
        select[sites + [nu+i for i in sites]] = 1
        projectors.append(mm(frame * select[None, :], inv_frame))
        blocks.append([s+i for i in sites])
        start += size
    for gate in synthesize_decoder(frame):
        gates.append((gate[0], *(int(q)+s for q in gate[1:])))
    return Recovery(n, s, stabilizers, gates, blocks, kernel, residual,
                    projectors, rank(affine), rank(features))
