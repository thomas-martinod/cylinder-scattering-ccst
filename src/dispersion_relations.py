import numpy as np
import matplotlib.pyplot as plt


# keep this if you have latex installed, else remove this preamble
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsfonts}"
})

# Physical parameters
rho = 1e5       # Density [kg/m³]
lam = 2.8e10    # Lambda [Pa]
eta = 1.62e9    # Eta (couple-stress modulus) [Pa·m²]
mu  = 4e9       # Shear modulus [Pa]

# Derived parameters
c1 = np.sqrt((lam + 2*mu) / rho)   # Longitudinal wave speed
c2 = np.sqrt(mu / rho)             # Shear (transverse) wave speed
ell = np.sqrt(eta / mu)            # Characteristic length

print(f"c1 = {c1:.2e} m/s,  c2 = {c2:.2e} m/s,  ℓ = {ell:.2e} m")

# Dimensionless wavenumber range (kℓ)
kL = np.linspace(0, 4, 400)
k = kL / ell  # Dimensional wavenumber (not used in plots but useful for reference)

# Dimensionless dispersion relations: ωℓ / c₂
omega_P_star = (c1 / c2) * kL                  # Longitudinal (P) wave
omega_S_star = kL * np.sqrt(1 + kL**2)         # Transverse (S) wave under C-CST

# Dimensionless phase and group velocities: v/c₂ and g/c₂
vP_star = np.full_like(kL, c1 / c2)            # P-wave (constant)
gP_star = np.full_like(kL, c1 / c2)            # P-wave group velocity (same)

vS_star = np.sqrt(1 + kL**2)                   # S-wave phase velocity
gS_star = (1 + 2*kL**2) / np.sqrt(1 + kL**2)   # S-wave group velocity

# Plot 1 — Dimensionless dispersion relations
plt.figure(figsize=(7, 5))
plt.plot(kL, omega_P_star, 'r-', lw=2, label=r'$\omega_P \ell / c_2 = (c_1/c_2)(k\ell)$')
plt.plot(kL, omega_S_star, 'b-', lw=2, label=r'$\omega_S \ell / c_2 = k\ell\sqrt{1+(k\ell)^2}$')
plt.xlabel(r'$k\ell$')
plt.ylabel(r'$\omega\ell / c_2$')
plt.title('Dimensionless Dispersion Relations for P and S Waves')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Plot 2 — Dimensionless phase and group velocities
plt.figure(figsize=(7, 5))
plt.plot(kL, vP_star, 'r--', lw=2, label=r'$v_P/c_2 = g_P/c_2 = c_1/c_2$')
plt.plot(kL, vS_star, 'b-', lw=2, label=r'$v_S/c_2 = \sqrt{1+(k\ell)^2}$')
plt.plot(kL, gS_star, 'g-', lw=2, label=r'$g_S/c_2 = (1+2(k\ell)^2)/\sqrt{1+(k\ell)^2}$')
plt.xlabel(r'$k\ell$')
plt.ylabel(r'$v/c_2,\; g/c_2$')
plt.title('Dimensionless Phase and Group Velocities for P and S Waves')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.show()
