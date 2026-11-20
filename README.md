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
| 5a | Chebyshev representation of the amplitude (moments in U_n(z)), BLAS-based kernel | M_N independent of N_cheb; = Nyström | ✅ N_cheb=4: 1.1788 GeV, 6 s/solve |
| 5 | Canonical normalisation of Ψ via F₁ᵖ(0) = 1 (EM current, 6 diagrams) | — | todo |
| 6 | Graviton–nucleon vertex, GFFs A, J, D, c̄ | paper figures | todo |

## Run

```bash
pip install numpy scipy
python3 scripts/01_quark_propagator.py
python3 scripts/02_diquark_normalization.py
python3 scripts/03_faddeev_nucleon.py          # ~15 s
python3 scripts/04_chebyshev_amplitude.py      # ~1 min
python3 scripts/05_em_current_one_loop.py      # ~40 s
python3 scripts/06_em_current_full.py coarse   # ~12 min (fine: ~1.5 h)
python3 scripts/07_gravitational_ff.py coarse  # ~1 h
python3 scripts/08_compare_gff.py <reference.json> coarse   # side-by-side with reference data
```

## Layout

- `qdq/dirac.py` — Euclidean γ-matrices, C = γ₂γ₄, γ₅ = γ₄γ₁γ₂γ₃ (+ self-checks)
- `qdq/quark.py` — S(p), σ_V, σ_S, M(p²); entire functions, so complex momenta are fine
- `qdq/quadrature.py` — 4D Euclidean loop quadrature (Gauss–Legendre radial × Gauss–Chebyshev angle)
- `qdq/diquark.py` — Γ^{0+}, Γ^{1+}, Δ^{J^P}, normalisation condition
- `qdq/current.py` — EM vertices (quark, 0⁺, 1⁺, 0⁺↔1⁺), flavour–charge factors, diagrams 1–3 in the Breit frame, F₁/F₂/F₃ trace projection, normalisation via dλ/dM, quark-loop inside the diquark
- `qdq/probes.py` — probe interface: photon and graviton vertices, flavour factors, seagull lifting, projectors (F₁/F₂/F₃ and A/J/D/S/c̄)
- `qdq/exchange.py` — two-loop diagrams 4–6: gauged Faddeev kernel (photon on the exchanged quark + seagulls), BLAS-factorised double integral
- `qdq/faddeev.py` — nucleon Faddeev kernel (Alkofer et al. Eqs. 8–13), flavour coefficients, positive-energy basis & projection, eigenvalue/mass solver; amplitude as Chebyshev moments f_{c,n}(|ℓ|) with `scalar_functions(l, P)` for arbitrary (also boosted) P

## Conventions worth knowing

- Rest frame P = (0,0,0,iM), Λ₊ = (1+γ₄)/2; quark momentum ℓ + P/3, diquark −ℓ + 2P/3.
- With our γ-matrix representation, the canonically normalised g₁₊ is purely imaginary; the phase
  must be kept (g_X g_Y enters the kernel). Dropping it produces spurious complex eigenvalue pairs.
- Kernel prefactor −4 = 2 (coupling to Ψ₁, Ψ₂) × (−2) (colour). The coupling values printed in
  2611.19819 are 2× ours — only the product with the kernel prefactor is physical.

## Amplitude: what matters (N_cheb study, scripts/04)

- Leading structures: s₁ (1.00), a₃ ≈ a₅ (0.58 each, i.e. mostly γ₅γ^⊥_ν), a₂ (0.27), a₄, a₁ (≈0.2), s₂ (0.17), a₆ (0.1).
- Mass: N_cheb = 2 already within 0.5 %, N_cheb = 4 within 0.01 %.
- Pointwise amplitude: a₁, a₄, a₆ need N_cheb = 6 for ~2 % accuracy (8 % at N_cheb = 4) → use 6 for form factors.

## EM current: what is established (scripts/05)

- Normalisation without two-loop diagrams: the Ward identity gives the full baryon-number charge as
  J^B = −(∂λ/∂P) Ψ̄G₀Ψ (λ = Faddeev eigenvalue). Diagrams 1+2 with baryon charges reproduce the
  independent finite difference −Ψ̄(∂G₀/∂P)Ψ to 7 digits; they carry 80.3 % of the baryon number,
  so diagrams 4–6 must carry 19.7 %.
- Normalised one-loop results: F₁ᵖ(0) = 0.817, F₁ⁿ(0) = −0.014 ⇒ diagrams 4–6 must give
  +0.183 (p) and +0.014 (n). Diagram 3 is purely isovector in F₂ (±0.43 at small Q²).
- 0⁺↔1⁺ transition vertex: with g₁₊ imaginary in our γ representation, the phase of κ₀₁ and the
  relative sign of χ¹⁰ vs χ⁰¹ are representation dependent. Taking 2507.13484's "χ¹⁰ = −χ⁰¹"
  literally makes the two terms cancel exactly. We derive them from the quark loop inside the
  diquark (same conventions as kernel and normalisation; its elastic 0⁺ limit reproduces 2K_μ):
  κ₀₁ = κ₁₀ = −2.39 i ⇒ use χ¹⁰ = +χ⁰¹, phase −i, magnitude 2.64 from the paper.
- Accuracy: integration grid converged to 4–5 digits; Faddeev resolution affects x = Q²/M² ≤ 1 at
  0.1 %, but F₂ at x = 3 by 6 % (N_cheb 6 → 8): Chebyshev continuation outside [−1, 1] degrades with Q².

## Full EM current (scripts/06, coarse two-loop grid) vs Cheng et al. 2507.13484 Table 2

Normalisation from dλ/dM only (no fit): baryon number 1.0008, F₁ᵖ(0) = 1.0002, F₁ⁿ(0) = 0.0005.

| | this code | 2507.13484 |
|---|---|---|
| μ_p | 2.71 | 2.80 |
| μ_n | −1.78 | −1.86 |
| r_E^p | 0.68 fm | 0.67 fm |
| r_E^n² | −0.088 fm² (−0.30²) | −0.084 fm² (−0.29²) |
| r_M^p | 0.64 fm | 0.67 fm |
| r_M^n | 0.69 fm | 0.72 fm |

Form factors on x = Q²/M² ≤ 3: `results/em_form_factors_coarse.json` (per-diagram breakdown included).
Static properties use G_E(0) computed on the same grid (the log's section e predates that fix).

Fine two-loop grid check (results/run_fine_partial.log, stopped after x = 0.02): F₁ᵖ(0) = 0.9996,
F₁ⁿ(0) = 0.0001, baryon number 0.9997; G_M changes by 1e-4 vs the coarse grid, so the 3–5 % gap
to 2507.13484 in the magnetic moments is not a two-loop discretisation effect.

## Gravitational form factors (scripts/07, coarse two-loop grid; results/gff_coarse.json)

Compared point by point with reference data from the authors of 2611.19819 (not stored here;
`scripts/08_compare_gff.py` reads them from a JSON file):

- **A(Q²)** and all its breakdowns (s/a/m diquark content, u/d) agree to ≲ 0.5 % for Q² ≲ 1.4 GeV²
  (−8 % at 4 GeV²: Chebyshev continuation of the boosted amplitude). A(0) = 1.004, A^s/A^a/A^m(0) =
  53.5/33.9/13.0 %, A^d/A^u(0) = 0.456 (hadron scale; the paper's table value 0.529 is at 2 GeV with sea).
- **D(Q²)** and breakdowns agree to ≲ 1 % for Q² ≳ 0.3 GeV² (≈ 0.1 % at 0.7–1.4 GeV²). At small Q²,
  J and D are extracted with a 1/τ factor and the coarse two-loop grid makes them noisy there.
- **c̄ (EMT conservation)**: our two-loop/seagull construction conserves exactly — with r_sc = r_av = 0,
  Σc̄ = 1e-4 at all Q². With the diquark form factor F_J(Q²) multiplying the whole diquark vertex
  (literal Eqs. 21, 23) conservation is broken (Σc̄ → −0.08); applying F_J only to the l_ν × (EM)
  part gives +0.01. The reference has Σc̄ ≈ 1e-4 with A matching the physical radii, so F_J must act
  only on parts of the vertex transverse to Q in the reference code — to be confirmed.
- **J(Q²)** is ≈ 4 % low (J(0) ≈ 0.479 vs ½), the deficit sitting in the scalar-diquark sector and in
  J^d; J^u agrees. Not from amplitude resolution (N_cheb up to 10), one-loop grid, frame dependence
  (Breit vs rest frame: 0.3 %), the C-symmetry of the vertex (exact) or the diquark radii. J is
  carried almost entirely by the transverse quark-vertex terms (unconstrained by the WGTI), and all
  spin-flip observables are low by a similar factor: J(0) 0.96, S(0) 0.95, μ_p 0.97, μ_n 0.96 — a
  common origin in the spin-flip coupling is likely. Open: per-diagram reference numbers would
  localise it.
