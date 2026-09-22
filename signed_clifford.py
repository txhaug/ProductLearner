"""Signed Clifford algebra and synthesis over binary Pauli labels.

Labels are rows ``(x_0,...,x_{n-1},z_0,...,z_{n-1})`` and denote the
Hermitian operators ``W(x,z) = i**(x.dot(z)) X**x Z**z``.  Signs are integers
``+1`` or ``-1``.  Circuits are gate tuples in application order.  These
routines manipulate polynomial-size binary arrays; they never use a state
vector, enumerate Paulis, or search the Clifford group.
"""
import numpy as np

try:
    from .clifford_learning import (mm, nullspace, rank, rref, symplectic_basis,
                                    symplectic_form, synthesize_decoder,
                                    transform_labels)
except ImportError:
    from clifford_learning import (mm, nullspace, rank, rref, symplectic_basis,
                                   symplectic_form, synthesize_decoder,
                                   transform_labels)


def _binary(a, ndim=None):
    raw = np.asarray(a)
    if ndim is not None and raw.ndim != ndim:
        raise ValueError(f'Expected a {ndim}-dimensional binary array')
    if np.any((raw != 0) & (raw != 1)):
        raise ValueError('Binary array entries must be zero or one')
    return raw.astype(np.uint8, copy=True)


def _signs(signs, length):
    signs = np.asarray(signs)
    if signs.shape != (length,) or np.any((signs != 1) & (signs != -1)):
        raise ValueError('Expected one +1/-1 sign per Pauli')
    return signs.astype(np.int8, copy=True)


def inverse_circuit(gates):
    """Return the inverse Clifford circuit, up to an irrelevant global phase."""
    inverse = {'H': 'H', 'S': 'SDG', 'SDG': 'S', 'CX': 'CX',
               'SWAP': 'SWAP', 'X': 'X', 'Y': 'Y', 'Z': 'Z'}
    result = []
    for gate in reversed(gates):
        if gate[0] not in inverse:
            raise ValueError(f'Unsupported Clifford gate {gate}')
        result.append((inverse[gate[0]], *gate[1:]))
    return result


def conjugate_paulis(labels, signs, gates):
    """Return the signed images ``C (sign * W_label) C^dagger``.

    Neither the input arrays nor the circuit is changed.  H, S, SDG, CX,
    SWAP and the three single-qubit Pauli gates are supported.  Keeping the
    signs is essential: e.g. H sends Y to -Y, and S sends Y to -X.
    """
    rows = _binary(labels, 2)
    if rows.shape[1] % 2:
        raise ValueError('A Pauli label must have an even number of entries')
    result_signs = _signs(signs, len(rows))
    n = rows.shape[1] // 2
    for gate in gates:
        kind, *sites = gate
        arity = 2 if kind in ('CX', 'SWAP') else 1
        if (len(sites) != arity or any(not isinstance(q, (int, np.integer))
                                     or q < 0 or q >= n for q in sites)
                or len(set(sites)) != len(sites)):
            raise ValueError(f'Invalid gate sites: {gate}')
        a = sites[0]
        x, z = rows[:, a], rows[:, n+a]
        if kind in ('H', 'S'):
            flip = x & z
        elif kind == 'SDG':
            flip = x & (z ^ 1)
        elif kind == 'CX':
            b = sites[1]
            flip = x & rows[:, n+b] & (rows[:, b] ^ z ^ 1)
        elif kind == 'SWAP':
            flip = np.zeros(len(rows), dtype=np.uint8)
        elif kind == 'X':
            flip = z.copy()
        elif kind == 'Z':
            flip = x.copy()
        elif kind == 'Y':
            flip = x ^ z
        else:
            raise ValueError(f'Unsupported Clifford gate {gate}')
        result_signs[flip.astype(bool)] *= -1
        if kind not in ('X', 'Y', 'Z'):
            transform_labels(rows, gate)
    return rows, result_signs


def multiply_paulis(label_a, sign_a, label_b, sign_b):
    """Multiply two commuting signed Hermitian Paulis.

    Return ``(label, sign)``.  Anticommuting operands have an imaginary phase
    and raise ValueError; silently dropping that phase would corrupt signs.
    """
    a, b = _binary(label_a, 1), _binary(label_b, 1)
    if a.shape != b.shape or len(a) % 2:
        raise ValueError('Pauli labels must have the same even length')
    sa, sb = map(int, _signs([sign_a, sign_b], 2))
    n = len(a) // 2
    c = a ^ b
    # Integer dot products, not GF(2) dot products: phases live modulo four.
    exponent = (int(np.sum(a[:n] & a[n:]))
                + int(np.sum(b[:n] & b[n:]))
                - int(np.sum(c[:n] & c[n:]))
                + 2*int(np.sum(a[n:] & b[:n]))) % 4
    if exponent % 2:
        raise ValueError('Cannot represent an anticommuting Pauli product by a real sign')
    return c, sa*sb*(-1 if exponent == 2 else 1)


def solve_binary(a, b):
    """Solve ``a @ x = b (mod 2)``, setting all free variables to zero.

    An inconsistent system raises ValueError.  The solution is unique when
    the matrix has full column rank; uniqueness is otherwise not assumed.
    """
    a, b = _binary(a, 2), _binary(b, 1)
    if len(b) != a.shape[0]:
        raise ValueError('Right-hand side has the wrong length')
    width = a.shape[1]
    reduced, pivots = rref(np.column_stack((a, b)))
    if width in pivots:
        raise ValueError('Inconsistent binary linear system')
    out = np.zeros(width, dtype=np.uint8)
    for row, pivot in zip(reduced, pivots):
        out[pivot] = row[-1]
    return out


def synthesize_clifford(frame, signs):
    """Synthesize the Clifford with the specified signed Pauli images.

    Column j of the 2n-by-2n symplectic frame is the image of X_j, and column
    n+j is the image of Z_j.  ``signs`` uses the same ordering.  The result is
    unique as an operator up to global phase.  Synthesis uses the unsigned
    decoder and then corrects its Pauli-image signs.
    """
    frame = _binary(frame, 2)
    size = frame.shape[0]
    if frame.shape != (size, size) or size % 2:
        raise ValueError('Expected an even-dimensional square symplectic frame')
    signs = _signs(signs, size)
    n = size // 2
    j = symplectic_form(n)
    if not np.array_equal(mm(frame.T, mm(j, frame)), j):
        raise ValueError('Frame is not symplectic')
    decoder = synthesize_decoder(frame)
    decoded, decoded_signs = conjugate_paulis(frame.T, signs, decoder)
    if not np.array_equal(decoded, np.eye(size, dtype=np.uint8)):
        raise ValueError('Unsigned Clifford synthesis produced the wrong frame')
    for q in range(n):
        if decoded_signs[q] == -1:
            decoder.append(('Z', q))
        if decoded_signs[n+q] == -1:
            decoder.append(('X', q))
    return inverse_circuit(decoder)


def extend_commuting_axes(labels, signs):
    """Synthesize A with ``A Z_j A^dagger = signs[j] W_labels[j]``.

    The t supplied axes must be independent and mutually commuting.  The
    remaining Pauli images are completed by binary linear algebra.  Each
    supplied generator is preserved individually (not just their span).
    """
    labels = _binary(labels, 2)
    t, size = labels.shape
    if size % 2:
        raise ValueError('Pauli labels must have even length')
    n = size // 2
    signs = _signs(signs, t)
    j = symplectic_form(n)
    if t > n or rank(labels) != t or np.any(mm(labels, mm(j, labels.T))):
        raise ValueError('Axes must be independent and mutually commuting')
    xs = []
    constraints = mm(labels, j)
    for i in range(t):
        target = np.zeros(t, dtype=np.uint8)
        target[i] = 1
        x = solve_binary(constraints, target)
        # Adding Z_j preserves all X_i/Z_k pairings and removes X_i/X_j
        # pairings, without changing previously enforced pairings.
        for k, previous in enumerate(xs):
            if int(mm(x @ j, previous)):
                x ^= labels[k]
        xs.append(x)
    paired = np.asarray(xs + list(labels), dtype=np.uint8).reshape(2*t, size)
    complement = nullspace(mm(paired, j))
    extra_xs, extra_zs = symplectic_basis(complement, j)
    frame = np.asarray(xs + extra_xs + list(labels) + extra_zs,
                       dtype=np.uint8).reshape(size, size).T
    frame_signs = np.ones(size, dtype=np.int8)
    frame_signs[n:n+t] = signs
    return synthesize_clifford(frame, frame_signs)
