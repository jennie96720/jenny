"""Euclidean Dirac algebra.

Conventions (as in Cloet/Segovia/Roberts papers):
  {gamma_mu, gamma_nu} = 2 delta_{mu nu},  gamma_mu^dagger = gamma_mu,
  a.b = sum_{i=1}^4 a_i b_i,  a timelike on-shell vector has P^2 = -m^2,
  gamma_5 = gamma_4 gamma_1 gamma_2 gamma_3,  C = gamma_2 gamma_4.
"""
import numpy as np

s0 = np.eye(2, dtype=complex)
s1 = np.array([[0, 1], [1, 0]], dtype=complex)
s2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
s3 = np.array([[1, 0], [0, -1]], dtype=complex)
Z2 = np.zeros((2, 2), dtype=complex)

def _block(a, b, c, d):
    return np.block([[a, b], [c, d]])

# gamma_k = [[0, -i sigma_k], [i sigma_k, 0]],  gamma_4 = diag(1, -1)
G = np.array([_block(Z2, -1j * s, 1j * s, Z2) for s in (s1, s2, s3)]
             + [_block(s0, Z2, Z2, -s0)])           # shape (4,4,4): G[mu]
I4 = np.eye(4, dtype=complex)
G5 = G[3] @ G[0] @ G[1] @ G[2]
C = G[1] @ G[3]
Cdag = C.conj().T

def slash(p):
    """gamma.p for p with shape (..., 4) (complex allowed) -> (..., 4, 4)."""
    return np.einsum('...m,mab->...ab', p, G)

def check():
    for m in range(4):
        for n in range(4):
            ac = G[m] @ G[n] + G[n] @ G[m]
            assert np.allclose(ac, 2 * (m == n) * I4)
    assert np.allclose(G5 @ G5, I4)
    # charge conjugation: C gamma_mu^T C^dagger = -gamma_mu
    for m in range(4):
        assert np.allclose(C @ G[m].T @ Cdag, -G[m])
    return True

if __name__ == "__main__":
    print("Dirac algebra checks:", check())
