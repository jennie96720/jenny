# jenny — quark + diquark Faddeev equation from scratch

Goal: reproduce, step by step, the quark+diquark framework of
P. Cheng, Z.-N. Xu, D.-Y. Fu, C. Chen, Z.-Q. Yao, *Quark + diquark description of nucleon
gravitational form factors*, arXiv:2611.19819 (App. A), which builds on
Cloët et al. arXiv:0812.0416, Segovia et al. arXiv:1408.2919, Alkofer et al. nucl-th/0412046.

## Roadmap

| # | Milestone | Check against | Status |
|---|-----------|---------------|--------|
| 1 | Algebraic dressed-quark propagator (A1–A4) | m_q = 5.08 MeV, M^E = 0.33 GeV | ✅ 5.08 MeV, 0.328 GeV |
| 2 | Diquark amplitudes + canonical normalisation (A5–A12) | g₀₊ = 14.94, g₁₊ = 12.98 | ✅ 2×(7.54, 6.51) = 15.08 / 13.02; factor 2 is bookkeeping, fixed by step 4 |
| 3 | Faddeev kernel: quark exchange, colour–flavour coefficients, 8 scalar functions s₁,₂, a₁…₆ | c_ss=½, c_sa=c_as=−√3/2, c_aa=−½ | ✅ |
| 4 | Solve Faddeev eigenvalue problem (Nyström on (\|ℓ\|, z), rest frame P = (0,0,0,i m_N)) | m_N = 1.18 GeV; 0⁺-only 1.46 GeV | ✅ 1.174 / 1.455 GeV |
| 5 | Canonical normalisation of Ψ via F₁ᵖ(0) = 1 (EM current, 6 diagrams) | — | todo |
| 6 | Graviton–nucleon vertex, GFFs A, J, D, c̄ | paper figures | todo |

## Run

```bash
pip install numpy scipy
python3 scripts/01_quark_propagator.py
python3 scripts/02_diquark_normalization.py
python3 scripts/03_faddeev_nucleon.py          # ~4 min on grid (16,8,8)
```

## Layout

- `qdq/dirac.py` — Euclidean γ-matrices, C = γ₂γ₄, γ₅ = γ₄γ₁γ₂γ₃ (+ self-checks)
- `qdq/quark.py` — S(p), σ_V, σ_S, M(p²); entire functions, so complex momenta are fine
- `qdq/quadrature.py` — 4D Euclidean loop quadrature (Gauss–Legendre radial × Gauss–Chebyshev angle)
- `qdq/diquark.py` — Γ^{0+}, Γ^{1+}, Δ^{J^P}, normalisation condition
- `qdq/faddeev.py` — nucleon Faddeev kernel (Alkofer et al. Eqs. 8–13), flavour coefficients, positive-energy basis & projection, eigenvalue/mass solver

## Conventions worth knowing

- Rest frame P = (0,0,0,iM), Λ₊ = (1+γ₄)/2; quark momentum ℓ + P/3, diquark −ℓ + 2P/3.
- With our γ-matrix representation, the canonically normalised g₁₊ is purely imaginary; the phase
  must be kept (g_X g_Y enters the kernel). Dropping it produces spurious complex eigenvalue pairs.
- Kernel prefactor −4 = 2 (coupling to Ψ₁, Ψ₂) × (−2) (colour). The coupling values printed in
  2611.19819 are 2× ours — only the product with the kernel prefactor is physical.
