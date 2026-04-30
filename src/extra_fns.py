# extra_fns.py
# by thomas martinod

import sympy as smp
from continuum_mechanics import vector

'''
# ----------------------------------------------------------------------
# Undo SymPy's special-function derivative recurrence expansions
# ----------------------------------------------------------------------

# Pretty derivative placeholders (these print as J'(n,x), etc.)
Jprime  = smp.Function("J'")
Yprime  = smp.Function("Y'")
Iprime  = smp.Function("I'")
Kprime  = smp.Function("K'")
H1prime = smp.Function("H^1'")   # Hankel 1st kind derivative placeholder
H2prime = smp.Function("H^2'")   # Hankel 2nd kind derivative placeholder

# Wildcards for pattern matching
_m = smp.Wild("m", exclude=[smp.I])
_x = smp.Wild("x")

def _undo_pair(expr, f_nm1, f_np1, repl_fn):
    """
    Replace recurrence-derivative patterns of the form:
        + f_{n-1}(x)/2 - f_{n+1}(x)/2
    and its reordered equivalent:
        - f_{n+1}(x)/2 + f_{n-1}(x)/2
    with repl_fn(n, x).
    """
    pat1 = f_nm1/2 - f_np1/2
    pat2 = -f_np1/2 + f_nm1/2
    expr = expr.replace(pat1, repl_fn(_m, _x))
    expr = expr.replace(pat2, repl_fn(_m, _x))
    return expr

def _undo_sum(expr, f_nm1, f_np1, repl_fn, overall_sign=+1):
    """
    Replace sum-type recurrence derivative patterns:
        overall_sign*( f_{n-1}(x)/2 + f_{n+1}(x)/2 )
    and its reordered equivalent.
    """
    pat1 = overall_sign*(f_nm1/2 + f_np1/2)
    pat2 = overall_sign*(f_np1/2 + f_nm1/2)
    expr = expr.replace(pat1, repl_fn(_m, _x))
    expr = expr.replace(pat2, repl_fn(_m, _x))
    return expr

def undo_special_derivs(expr):
    """
    Undo SymPy's automatic derivative expansions for:
      J_n, Y_n, I_n, K_n, Hankel1_n, Hankel2_n

    Maps:
      (J_{n-1}(x) - J_{n+1}(x))/2  -> J'(n,x)
      (Y_{n-1}(x) - Y_{n+1}(x))/2  -> Y'(n,x)
      (H1_{n-1}(x) - H1_{n+1}(x))/2 -> H1'(n,x)
      (H2_{n-1}(x) - H2_{n+1}(x))/2 -> H2'(n,x)
      (I_{n-1}(x) + I_{n+1}(x))/2  -> I'(n,x)
      -(K_{n-1}(x) + K_{n+1}(x))/2 -> K'(n,x)
    """
    # Build the matched function atoms with wildcards
    J_nm1 = smp.besselj(_m - 1, _x); J_np1 = smp.besselj(_m + 1, _x)
    Y_nm1 = smp.bessely(_m - 1, _x); Y_np1 = smp.bessely(_m + 1, _x)
    H1_nm1 = smp.hankel1(_m - 1, _x); H1_np1 = smp.hankel1(_m + 1, _x)
    H2_nm1 = smp.hankel2(_m - 1, _x); H2_np1 = smp.hankel2(_m + 1, _x)
    I_nm1 = smp.besseli(_m - 1, _x); I_np1 = smp.besseli(_m + 1, _x)
    K_nm1 = smp.besselk(_m - 1, _x); K_np1 = smp.besselk(_m + 1, _x)

    # Difference-type: (f_{n-1}-f_{n+1})/2
    expr = _undo_pair(expr, J_nm1, J_np1, Jprime)
    expr = _undo_pair(expr, Y_nm1, Y_np1, Yprime)
    expr = _undo_pair(expr, H1_nm1, H1_np1, H1prime)
    expr = _undo_pair(expr, H2_nm1, H2_np1, H2prime)

    # Sum-type:
    # I_n'(x)  = (I_{n-1}+I_{n+1})/2
    expr = _undo_sum(expr, I_nm1, I_np1, Iprime, overall_sign=+1)

    # K_n'(x)  = -(K_{n-1}+K_{n+1})/2
    expr = _undo_sum(expr, K_nm1, K_np1, Kprime, overall_sign=-1)

    return expr
'''

# ---------------------------------------------------------
# function for displaying latex notation for copying and pasting
def latex_print(expr):
    print(smp.latex(expr))

# ---------------------------------------------------------
# function that computes the grad of a scalar function in generalized coordinates
def grad(u, coords, h_vec):
    """
    compute the gradient of a scalar function phi.

    parameters:
    ----------
    u : sympy expression
        scalar function to compute the gradient from.
    coords : Tuple (3)
        coordinates for the reference system. As an example, for cylindrical coordinates
        one can input coords = (r, theta, z), where each component is a symbol
    h_vec : Tuple (3)
        scale coefficients for the new coordinate system. For the cyilndrical coordinates
        case, h_vec = (1, r, 1)

    returns
    -------
    gradient: Matrix (3, 1)
        Column vector with the components of the gradient.
    """
    return smp.Matrix(3, 1, lambda i, j: u.diff(coords[i])/h_vec[i])


# ---------------------------------------------------------
# function that computes the vector laplacian of a scalar function in generalized coordinates
def lap_vec(A, coords, h_vec):
    """
    Laplacian of a vector function A.

    Parameters
    ----------
    A : Matrix, List
        Vector function to compute the laplacian from.
    coords : Tuple (3), optional
        Coordinates for the new reference system. It takes (x, y, z) 
        as default.
    h_vec : Tuple (3), optional
        Scale coefficients for the new coordinate system. It takes
        (1, 1, 1), as default.

    Returns
    -------
    laplacian : Matrix (3, 1)
        Column vector with the components of the Laplacian.
    """
    return grad(vector.div(A, coords, h_vec), coords, h_vec) \
        - vector.curl(vector.curl(A, coords, h_vec), coords, h_vec)

# ---------------------------------------------------------
# function that computes the rotation vector and rotation tensor of a certain displacement field:
def rotation(u, coords, hvec):
    """
    Rotation vector and tensor of a displacement vector u

    Parameters
    ----------
    u : Matrix, List
        Displacement vector function of the coords to find the rotation 
        vector and tensor of
    coords : Tuple (3), optional
        Coordinates for the new reference system. It takes (x, y, z) 
        as default.
    h_vec : Tuple (3), optional
        Scale coefficients for the new coordinate system. It takes
        (1, 1, 1), as default.

    Returns
    -------
    w_vector : Matrix (3, 1)
        Rotation column vector
    
    w_tensor : Matrix (3,3)
        Rotation tensor skew-symmetric matrix dual of w_vector 
    """
    w_vector = smp.Rational(1,2) * vector.curl(u, coords, hvec)
    w_tensor = vector.dual_tensor(w_vector)
    return w_vector, w_tensor


# ---------------------------------------------------------
# function to compute the symmetric and skew symmetric stress tensor, the stress tensor 
# and the couple stress tensor in C-CST
def complete_strain_stress_ccst(strain, curvature, rotation_tensor, coords, h_vec, parameters):
    """
    Return the force-stress and couple-stress tensor for
    given strain and curvature tensors and material 
    parameters for the Consistent Couple-Stress Theory
    (C-CST).
    
    Parameters
    ----------
    strain : Matrix (3, 3)
        Strain tensor.
    curvature : Matrix (3, 3)
        Curvature tensor.
    rotation_tensor : Matrix (3,3)
        Rotation tensor.
    coords : Tuple (3), optional
        Coordinates for the new reference system. It takes (x, y, z) 
        as default.
    h_vec : Tuple (3), optional
        Scale coefficients for the new coordinate system. It takes
        (1, 1, 1), as default.
    parameters : tuple
        Material parameters in the following order:

        lamda : float
            Lamé's first parameter.
        mu : float, > 0
            Lamé's second parameter.
        eta : float, > 0
            Couple stress modulus in C-CST.
    """
    lamda, mu, eta = parameters
    strain_trace = strain.trace()

    force_stress_symm = smp.Matrix(3, 3, lambda i, j:
                        lamda*smp.eye(3)[i, j]*strain_trace + 2*mu*strain[j, i])
    
    force_stress_skew = smp.Matrix(3, 3, lambda i, j: 2 * eta * 
                                   vector.lap(rotation_tensor[i,j], coords, h_vec))

    force_stress = force_stress_symm + force_stress_skew
    
    couple_stress = -8*eta*curvature
    return force_stress, force_stress_symm, force_stress_skew, couple_stress



def classical_limit_coeff(expr, beta1, beta2, kS, n, a):
    """
    Classical (Cauchy) limit of a C-CST scattered-field coefficient.

    As ell -> 0:  beta1 -> inf,  beta2 -> k_S.

    CASE A: K only in denominator  [Bn-type]
        → field contribution Bn · K_n(beta1·r) → 0
        → return 0

    CASE B: K in both num and den  [Cn-type]
        → 0/0 form
        → step 1: homogenize indices K_{n±1} → K_n  (asymptotic equivalence)
        → step 2: replace K_n and beta1 with dummy algebraic symbols t, b1
        → step 3: cancel t = K_n algebraically iteratively
        → step 4: take limit b1 → inf  (now just rational, fast and clean)
        → step 5: substitute beta2 → kS

    CASE C: no K anywhere
        → substitute beta2 → kS directly

    CASE D: K only in numerator  [unexpected physically]
        → warn and return 0

    Parameters
    ----------
    expr  : SymPy expression  (Bn_sln or Cn_sln)
    beta1 : SymPy symbol      (evanescent wavenumber)
    beta2 : SymPy symbol      (propagating wavenumber)
    kS    : SymPy symbol      (classical shear wavenumber)
    n     : SymPy symbol      (azimuthal order)
    a     : SymPy symbol      (cylinder radius)
    """

    def _has_K_beta1(e):
        return any(
            node.func is smp.besselk and node.args[1].has(beta1)
            for node in smp.preorder_traversal(e)
        )

    num, den = smp.fraction(smp.together(expr))
    K_in_num = _has_K_beta1(num)
    K_in_den = _has_K_beta1(den)

    # ── CASE A: K only in denominator → field vanishes ───────────────────────
    if K_in_den and not K_in_num:
        return smp.S.Zero

    # ── CASE D: unexpected ────────────────────────────────────────────────────
    if K_in_num and not K_in_den:
        print("WARNING: K in numerator but not denominator — unexpected case")
        return smp.S.Zero

    # ── CASE B: K in both → asymptotic cancellation ──────────────────────────
    if K_in_num and K_in_den:
        Kn   = smp.besselk(n,   beta1 * a)
        Kn_m = smp.besselk(n-1, beta1 * a)
        Kn_p = smp.besselk(n+1, beta1 * a)

        # step 1: homogenize K indices: K_{n±1} → K_n
        e = expr.subs([(Kn_m, Kn), (Kn_p, Kn)])

        # step 2: replace K_n and beta1 with dummy algebraic symbols
        t  = smp.Symbol('t',  positive=True)  # stands for K_n(beta1*a)
        b1 = smp.Symbol('b1', positive=True)  # stands for beta1
        e = e.subs([(Kn, t), (beta1, b1)])

        # step 3: cancel ALL factors of t iteratively
        for _ in range(10):
            num_e, den_e = smp.fraction(smp.together(e))
            if num_e.has(t) and den_e.has(t):
                e = smp.cancel(e / t)
            else:
                break

        # sanity check
        _, den_e = smp.fraction(smp.together(e))
        if den_e.has(t):
            print("WARNING: t did not cancel completely — check expression")

        # step 4: limit b1 → inf (now just a rational function, no Bessel)
        e = smp.limit(e, b1, smp.oo)

        # step 5: substitute beta2 → kS and simplify
        return smp.simplify(e.subs(beta2, kS))

    # ── CASE C: no K anywhere → direct substitution ──────────────────────────
    return smp.simplify(expr.subs(beta2, kS))

'''
def classical_limit_system(A_sym, b_sym, beta1, beta2, kS, eta, var_names):
    """
    Classical limit of A·x = b:
      1. beta2 → kS
      2. beta1 → inf  (K_n evanescent terms vanish)
      3. eta   → 0    (couple stress disappears: eta = mu*ell^2, ell→0)
      4. remove zero columns (those unknowns = 0)
      5. remove zero rows    (trivially satisfied equations)
      6. solve reduced system
      7. reconstruct full solution
    """
    # step 1-2: wavenumber limits
    A_cl = A_sym.subs(beta2, kS)
    b_cl = b_sym.subs(beta2, kS)
    A_cl = A_cl.applyfunc(lambda e: smp.limit(e, beta1, smp.oo))
    b_cl = b_cl.applyfunc(lambda e: smp.limit(e, beta1, smp.oo))

    # step 3: couple stress vanishes
    A_cl = A_cl.subs(eta, 0)
    b_cl = b_cl.subs(eta, 0)

    # step 4: find zero columns → those unknowns are 0
    n_cols = A_cl.shape[1]
    zero_cols, live_cols = [], []
    for j in range(n_cols):
        if all(smp.simplify(e) == 0 for e in A_cl.col(j)):
            zero_cols.append(j)
        else:
            live_cols.append(j)

    # step 5: find non-zero rows → independent equations
    n_rows = A_cl.shape[0]
    live_rows = []
    for i in range(n_rows):
        row = list(A_cl.row(i)) + [b_cl[i]]
        if any(smp.simplify(e) != 0 for e in row):
            live_rows.append(i)

    print(f"Classical limit → zero coefficients : {[var_names[j] for j in zero_cols]}")
    print(f"Classical limit → live coefficients : {[var_names[j] for j in live_cols]}")
    print(f"Classical limit → active equations  : {live_rows}")

    # step 6: build and solve reduced system
    A_red = A_cl.extract(live_rows, live_cols)
    b_red = b_cl.extract(live_rows, [0])
    sol_red = smp.simplify(A_red.LUsolve(b_red))

    # step 7: reconstruct full solution
    sol_full = {}
    red_idx = 0
    for j, name in enumerate(var_names):
        if j in zero_cols:
            sol_full[name] = smp.S.Zero
        else:
            sol_full[name] = sol_red[red_idx]
            red_idx += 1

    return A_cl, b_cl, sol_full
'''

# ---------------------------------------------------------
# ---------------------------------------------------------
def classical_limit_system(A_sym, b_sym,
                            beta1_list, beta2_list, kS_list,
                            eta_list, var_names,
                            force_stop_after_zeros=False):
    """
    Classical limit of A·x = b for any number of materials.
 
      1. beta2_i → kS_i   (propagating wavenumbers, direct substitution)
      2. beta1_i → inf    (evanescent terms vanish, element-wise limit)
      3. eta_i   → 0      (couple stresses vanish for all materials)
      4. iteratively zero unknowns whose column is zero or diverges
      5. find non-zero rows
      6. select independent rows NUMERICALLY (fast — avoids symbolic rank)
      7. solve reduced system symbolically
      8. reconstruct full solution vector
 
    Parameters
    ----------
    beta1_list : symbol or list  e.g. beta1  or [beta1, beta1_p]
    beta2_list : symbol or list  e.g. beta2  or [beta2, beta2_p]
    kS_list    : symbol or list  e.g. k      or [k, k_p]
    eta_list   : symbol or list  e.g. eta    or [eta, eta_p]
    var_names  : list of str     e.g. ["Bn","Cn"] or ["An","Bn","Cn","Dn"]
    force_stop_after_zeros : bool (default False)
        If True, stop after identifying which coefficients vanish classically
        and return without attempting to solve the remaining system.
        Useful when the reduced system is large (e.g. 4x4) and symbolic
        solution would take too long. Returns None for sol_full in that case.
    """
    import numpy as np

    # normalize all inputs to lists
    if not isinstance(beta1_list, (list, tuple)): beta1_list = [beta1_list]
    if not isinstance(beta2_list, (list, tuple)): beta2_list = [beta2_list]
    if not isinstance(kS_list,    (list, tuple)): kS_list    = [kS_list]
    if not isinstance(eta_list,   (list, tuple)): eta_list   = [eta_list]
 
    A_cl = A_sym.copy()
    b_cl = b_sym.copy()
 
    # step 1: beta2_i → kS_i
    for b2, ks in zip(beta2_list, kS_list):
        A_cl = A_cl.subs(b2, ks)
        b_cl = b_cl.subs(b2, ks)
 
    # step 2: beta1_i → inf
    for b1 in beta1_list:
        A_cl = A_cl.applyfunc(lambda e: smp.limit(e, b1, smp.oo))
        b_cl = b_cl.applyfunc(lambda e: smp.limit(e, b1, smp.oo))
 
    # step 3: eta_i → 0
    for eta_sym in eta_list:
        A_cl = A_cl.subs(eta_sym, 0)
        b_cl = b_cl.subs(eta_sym, 0)
 
    # step 4: iteratively zero unknowns whose column is zero or diverges
    zero_cols = []
    live_cols = list(range(A_cl.shape[1]))
 
    changed = True
    while changed:
        changed  = False
        new_zero = []
        for j in live_cols:
            col     = [smp.simplify(A_cl[i, j]) for i in range(A_cl.shape[0])]
            is_zero = all(e == 0 for e in col)
            has_inf = any(e.has(smp.oo) or e.has(smp.zoo) for e in col)
            if is_zero or has_inf:
                new_zero.append(j)
        if new_zero:
            changed = True
            for j in new_zero:
                zero_cols.append(j)
                live_cols.remove(j)
            print(f"  Pass: zeroed {[var_names[j] for j in new_zero]}, "
                  f"remaining: {[var_names[j] for j in live_cols]}")
 
    print(f"Classical limit → zero : {[var_names[j] for j in zero_cols]}")
    print(f"Classical limit → live : {[var_names[j] for j in live_cols]}")
 
    # ── early exit ────────────────────────────────────────────────────────────
    if force_stop_after_zeros:
        print("force_stop_after_zeros=True — skipping symbolic solve.")
        print("Classically zero coefficients confirmed. "
              "Remaining coefficients require numerical evaluation per (n, params).")
        return A_cl, b_cl, None
 
    # step 5: find non-zero rows
    live_rows = []
    for i in range(A_cl.shape[0]):
        row = ([smp.simplify(A_cl[i, j]) for j in live_cols]
               + [smp.simplify(b_cl[i])])
        if any(e != 0 for e in row):
            live_rows.append(i)
 
    print(f"Classical limit → rows : {live_rows}")
 
    # step 6: select independent rows NUMERICALLY (fast)
    # symbolic rank with Bessel functions is extremely slow (minutes per call)
    # numerical rank at a generic test point is equivalent and takes microseconds
    n_unknowns    = len(live_cols)
    selected_rows = []
 
    if n_unknowns > 0:
        # build test substitution: n=1, all other free symbols → distinct floats
        _test_vals = {}
        for idx, s in enumerate(A_cl.free_symbols):
            _test_vals[s] = 1 if str(s) == 'n' else (1.3 + 0.4 * idx)
 
        try:
            A_num_test = np.array(
                A_cl.subs(_test_vals).evalf().tolist(),
                dtype=complex
            )
 
            A_growing = np.zeros((0, n_unknowns), dtype=complex)
            for i in live_rows:
                candidate   = A_num_test[i:i+1, live_cols]
                A_candidate = np.vstack([A_growing, candidate])
                if (np.linalg.matrix_rank(A_candidate)
                        > np.linalg.matrix_rank(A_growing)):
                    selected_rows.append(i)
                    A_growing = A_candidate
                if len(selected_rows) == n_unknowns:
                    break
 
        except Exception as exc:
            print(f"  [numeric rank fallback] {exc}")
            selected_rows = live_rows[:n_unknowns]
 
    print(f"Classical limit → selected rows : {selected_rows}")
 
    # step 7: solve reduced system symbolically
    A_red   = A_cl.extract(selected_rows, live_cols)
    b_red   = b_cl.extract(selected_rows, [0])
    sol_red = smp.simplify(A_red.LUsolve(b_red))
 
    # step 8: reconstruct full solution
    sol_full = {}
    red_idx  = 0
    for j, name in enumerate(var_names):
        if j in zero_cols:
            sol_full[name] = smp.S.Zero
        else:
            sol_full[name] = sol_red[red_idx]
            red_idx += 1
 
    return A_cl, b_cl, sol_full