# jenny — quark + diquark Faddeev equation from scratch

Goal: reproduce, step by step, the quark+diquark framework of
P. Cheng, Z.-N. Xu, D.-Y. Fu, C. Chen, Z.-Q. Yao, *Quark + diquark description of nucleon
gravitational form factors*, arXiv:2611.19819 (App. A), which builds on
Cloët et al. arXiv:0812.0416, Segovia et al. arXiv:1408.2919, Alkofer et al. nucl-th/0412046.

## Roadmap

| # | Milestone | Check against | Status |
|---|-----------|---------------|--------|
| 1 | Algebraic dressed-quark propagator (A1–A4) | m_q = 5.08 MeV, M^E = 0.33 GeV | ✅ 5.08 MeV, 0.328 GeV |
| 2 | Diquark amplitudes + canonical normalisation (A5–A12) | g₀₊ = 14.94, g₁₊ = 12.98 | ⚠️ 15.08 / 13.02 up to a convention factor (see script) |
| 3 | Faddeev kernel: quark exchange, colour–flavour coefficients, 8 scalar functions s₁,₂, a₁…₆ | — | todo |
| 4 | Solve Faddeev eigenvalue problem (Chebyshev in ℓ̂·P̂, rest frame P = (0,0,0,i m_N)) | m_N = 1.18 GeV | todo |
| 5 | Canonical normalisation of Ψ via F₁ᵖ(0) = 1 (EM current, 6 diagrams) | — | todo |
| 6 | Graviton–nucleon vertex, GFFs A, J, D, c̄ | paper figures | todo |

## Run

```bash
pip install numpy scipy
python3 scripts/01_quark_propagator.py
python3 scripts/02_diquark_normalization.py
```

## Layout

- `qdq/dirac.py` — Euclidean γ-matrices, C = γ₂γ₄, γ₅ = γ₄γ₁γ₂γ₃ (+ self-checks)
- `qdq/quark.py` — S(p), σ_V, σ_S, M(p²); entire functions, so complex momenta are fine
- `qdq/quadrature.py` — 4D Euclidean loop quadrature (Gauss–Legendre radial × Gauss–Chebyshev angle)
- `qdq/diquark.py` — Γ^{0+}, Γ^{1+}, Δ^{J^P}, normalisation condition
