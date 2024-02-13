import jax
import jax.numpy as jnp
from jax.scipy.linalg import lu_solve, qr, solve_triangular, ordqz

def solve_discrete_are_jax(a, b, q, r, e=None, s=None, balanced=True):
    """
    Solves the discrete-time algebraic Riccati equation (DARE) using JAX.

    Args:
        a: State matrix (shape (m, m))
        b: Input matrix (shape (m, n))
        q: State cost matrix (shape (m, m))
        r: Input cost matrix (shape (n, n))
        e: (Optional) State weighting matrix (shape (m, m))
        s: (Optional) State-input cross-term matrix (shape (m, n))
        balanced: (Optional) Whether to perform matrix balancing. Default: True

    Returns:
        x: Solution matrix (shape (m, m))
    """

    m, n = b.shape
    r_or_c = complex if jnp.iscomplexobj(b) else float

    # Matrix Pencil Formation
    H = jnp.zeros((2*m+n, 2*m+n), dtype=r_or_c)
    H[:m, :m] = a
    H[:m, 2*m:] = b
    H[m:2*m, :m] = -q
    H[m:2*m, m:2*m] = jnp.eye(m) if e is None else e.conj().T
    H[m:2*m, 2*m:] = 0. if s is None else -s
    H[2*m:, :m] = 0. if s is None else s.conj().T
    H[2*m:, 2*m:] = r

    J = jnp.zeros_like(H, dtype=r_or_c)
    J[:m, :m] = jnp.eye(m) if e is None else e
    J[m:2*m, m:2*m] = a.conj().T
    J[2*m:, m:2*m] = -b.conj().T

    # Balancing 
    if balanced:
        M = jnp.abs(H) + jnp.abs(J)
        M[jnp.diag_indices_from(M)] = 0.  # Avoid modifying diagonals
        sca, _ = jnp.linalg.eig(M)        # More general than matrix_balance
        sca = jnp.log2(jnp.abs(sca))      # Ensure real for comparisons
        s = jnp.round((sca[m:2*m] - sca[:m])/2)
        sca = 2 ** jnp.r_[s, -s, sca[2*m:]]
        H = H * sca[:, None] * jnp.reciprocal(sca)
        J = J * sca[:, None] * jnp.reciprocal(sca)

    # QR Factorization 
    q_of_qr, _ = qr(H[:, -n:], mode='economic')
    H = q_of_qr[:, n:].conj().T @ H[:, :2*m]
    J = q_of_qr[:, n:].conj().T @ J[:, :2*m]

    # Generalized Schur/QZ Decomposition
    out_str = 'real' if r_or_c == float else 'complex'
    _, _, _, _, _, u = ordqz(H, J, sort='iuc', overwrite_a=True, 
                             overwrite_b=True, check_finite=False, output=out_str)

    # Stable Subspace Extraction
    if e is not None:
        u, _ = qr(jnp.vstack((e @ u[:m, :m], u[m:, :m])), mode='economic')
    u00 = u[:m, :m]
    u10 = u[m:, :m]

    # Back-Substitution with LU
    up, ul, uu = lu_solve(u00, jnp.eye(m))  
    if 1/jnp.linalg.cond(uu) < jnp.spacing(1.):
        raise ValueError("Failed to find finite solution")

    x = solve_triangular(ul.conj().T, solve_triangular(uu.conj().T,
                                                     u10.conj().T, lower=True),
                         unit_diagonal=True).conj().T @ up.conj().T

    if balanced:
        x = x * sca[:m, None] * sca[:m] 

    return jnp.real_if_close((x + x.conj().T) / 2) 