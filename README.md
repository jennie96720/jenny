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
| 6 | Graviton–nucleon vertex, GFFs A, J, D, c̄ | paper figures | ✅ (see below) |
| 7 | Valence DFs: Mellin moments ⟨x^m⟩ from weighted forward currents (qdq/pdf.py) | u, d numbers; ⟨x⟩ = A^q(0) | ✅ step 1 (m ≤ 5–6) |
| 8 | Quark-in-diquark DFs = DA² (qdq/diquark_df.py); AO evolution (qdq/evolution.py); x-shapes (qdq/pdf_x.py) | EPJC 84:739 Table 2, 2505.10663 Table 4 | ✅ valence moments agree with realistic CSM |
| 9 | Helicity DFs: forward axial current (qdq/pdf_pol.py, scripts/13) | g_A; Cheng et al. PLB 844 (2023) 138074 | ✅ Δu/g_A, Δd/g_A agree |
| 10 | Gluon helicity, proton spin decomposition (qdq/spin.py, scripts/14) | COMPASS a₀; Cheng et al. 2023 | ✅ a_I^E(ζ_C) = 0.32–0.33 |
| 11 | Fortran port of the DF integrals (fortran/sources/zero_pdf, modes pdf / axial_table / helicity / da; post-processing scripts/15) | Python, digit by digit | ✅ converged run (fortran/runs/fine): moments agree to ≤ 7e-9; 10 + 6 min with 4 MPI processes |
| 12 | Proton leading-twist DA: f_N and first moments (qdq/nucleon_da.py, scripts/16) | RQCD 2019 lattice; Mezrag et al. 2018 | ⚠️ scalar-diquark part finite; longitudinal axial-diquark part log UV divergent → quoted vs cutoff |
| 13 | Λ, Σ⁰: flavour-resolved Faddeev solver with s quarks (qdq/baryon_faddeev.py, scripts/17) | SU(3) limit = nucleon; Chen et al. 2019 couplings and core masses | ✅ masses |
| 14 | Λ, Σ⁰ valence DFs: Mellin moments m ≤ 6, all six diagrams, asymmetric [us]/{us} DAs (qdq/baryon_pdf.py, scripts/18) | proton = NucleonPDFMoments (6e-12); SU(3) octet relations (1e-12); SCI 2505.10663 | ✅ coarse grid; higher moments / fine grid next |

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

## Learning / Fortran

- `docs/fortran_guide.md` — step-by-step guide for an independent Fortran implementation (pitfalls, checks)
- `docs/reference_values.md` — numbers to compare against at each step (`scripts/make_reference_values.py`)
- `fortran/` — complete Fortran program in the structure of the group's meson DSE/BSE code (CMake, namelist config, optional MPI); every stage reproduces the Python numbers digit by digit (`fortran/README.md`)

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

Fine two-loop grid at x = 0.02 (GFFs): J 0.4656 (coarse 0.4663) — the J deficit is not a grid
effect; A 0.9712 and D −2.47 now agree with the reference (0.9716, −2.46), i.e. the small-Q² D
discrepancy on the coarse grid was two-loop discretisation, amplified by the 1/τ in the projector.

## Valence-quark DFs: Mellin moments (scripts/09, qdq/pdf.py; results/pdf_moments_step1.json)

Same DF construction as EPJC 84 (2024) 739 and arXiv:2505.10663 (SCI), but the momentum-dependent
amplitude does not allow the light-front δ-function to be integrated analytically, so we compute
moments ⟨x^m⟩_q = Tr[n·J^(m)_q] / Tr[n·J^B], n = (0,0,1,i), where each Q = 0 number-current insertion
is weighted by (n·p_struck/n·P)^m (struck line: k_q in diagram 1, the diquark in diagram 2, the
exchanged quark in diagram 4, the differentiated leg in seagulls 5, 6). Normalisation from dλ/dM.

Rest frame, one-loop (48,32,24), coarse two-loop grid, |k| < kmax:

| m | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| ⟨x^m⟩_u | 2.0010 | 0.6888 | (0.2700) | (0.1180) | (0.0564) | (0.0288) | (0.0155) |
| ⟨x^m⟩_d | 1.0013 | 0.3140 | (0.1125) | (0.0456) | (0.0203) | (0.0097) | (0.0049) |

- m = 0: u, d numbers 2, 1 to 0.1 % (Ward identity; a non-trivial check of normalisation + all 5 diagrams).
- m = 1: equals the gravitational A^u(0), A^d(0) = 0.6893, 0.3144 (0.1 %; diagram 4 differs by 0.8 %,
  coarse two-loop grid). Momentum sum ⟨x⟩_u + ⟨x⟩_d = 1.003.
- m ≥ 2 in brackets: diagram 2 still uses the placeholder ⟨z^m⟩_{q/D} = 2^{-m}; step 2 (below) supplies the
  quark-in-diquark DF moments. Diagram 2 alone gives the diquark LF distribution ⟨y^m⟩_D (in the JSON).
- Imaginary parts ~1e-12. kmax = 10 vs 15 GeV agree to 1e-5 for m ≤ 5, ~1 % at m = 6; m ≥ 7 unstable.
- Frame independence (boost p₃ = 0.1, 0.2, 0.4 GeV): m ≤ 4 moments change by 0.1–0.3 %; higher m
  degrade with boost (Chebyshev continuation), so the rest frame is used.

### Step 2: quark-in-diquark DFs (scripts/10, qdq/diquark_df.py; results/pdf_moments_step2.json)

As in EPJC 84:739 App. B and 2505.10663 App. A: q^D(z) = n [φ_D(z)]², φ_D the leading-twist DA of the
0⁺ (γ₅γ·n projection) or 1⁺ (γ·n n_ν, longitudinal) diquark, computed from our diquark amplitudes and
quark propagator via Euclidean moments (rest frame, |k| < 20 GeV; ⟨ξ²,⁴,⁶⟩ stable to < 1e-3).

| | ⟨ξ²⟩ | ⟨ξ⁴⟩ | ⟨ξ⁶⟩ | φ fit: α, a₂, a₄ | ⟨z²⟩ ⟨z³⟩ ⟨z⁴⟩ of φ² |
|---|---|---|---|---|---|
| 0⁺ | 0.1901 | 0.0755 | 0.0393 | 1.61, −0.005, −0.025 | 0.2850 0.1775 0.1177 |
| 1⁺ | 0.1090 | 0.0307 | 0.0124 | 3.36, −0.010, +0.005 | 0.2653 0.1480 0.0861 |
| asymptotic 6x(1−x) | 0.2 | 0.0857 | 0.0476 | 1.5, 0, 0 | 0.2857 0.1786 0.1190 |

φ(x) = N[x(1−x)]^{α−1/2}[1 + a₂C₂^α + a₄C₄^α]: the 2-parameter form cannot reach the narrow 1⁺ DA
(⟨ξ⁴⟩/⟨ξ²⟩² too large for its width). Proton moments with these ⟨z^m⟩ (2- vs 3-parameter DA: < 0.2 %):

| m | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| ⟨x^m⟩_u(ζ_H) | 2.0010 | 0.6888 | 0.2752 | 0.1223 | 0.0588 | 0.0301 | 0.0161 |
| ⟨x^m⟩_d(ζ_H) | 1.0013 | 0.3140 | 0.1165 | 0.0489 | 0.0223 | 0.0107 | 0.0053 |

### Steps 3–4: AO evolution and x-dependence (scripts/11; results/pdf_evolved.json, pdf_valence.png)

AO equations of Lu et al. (arXiv:2203.00753, Eqs. 3) in Mellin space — exact per moment, so no
x-reconstruction is needed for moments at any scale. Coupling: process-independent charge
γ_m π/ln[K²/Λ²] (Cui et al. 2020). With ζ_H = 0.3266 GeV (nominal 0.331; the papers quote ±5 %) it gives
⟨x⟩_val = 0.477 at 2 GeV and 0.441 at 3.097 GeV, i.e. exactly the evolution of both papers.

Valence at ζ₃ = 3.097 GeV, % (EPJC 84:739 Table 2):

| | this work | SCI | realistic CSM [21] |
|---|---|---|---|
| ⟨x⟩_u, ⟨x²⟩_u, ⟨x³⟩_u | 30.35, 7.65, 2.45 | 30.2, 8.4, 3.1 | 30.4(1.3), 7.7(5), 2.5(2) |
| ⟨x⟩_d, ⟨x²⟩_d, ⟨x³⟩_d | 13.84, 3.24, 0.98 | 13.9, 3.7, 1.3 | 13.8(6), 3.2(2), 0.9(1) |
| u − d | 16.51, 4.41, 1.47 | 16.4, 4.7, 1.8 | 16.6(7), 4.5(3), 1.6(1) |

The momentum-dependent q(qq) amplitude reproduces the realistic CSM moments and is clearly softer than
the SCI at higher m (narrower hadron-scale DFs).

Evolution checks (feeding SCI hadron-scale moments): glue at 2 GeV 41.2, 2.70, 0.510 % (n = 1, 2, 3) vs
41.4, 2.67, 0.500 % in 2505.10663 Table 4; total sea 10.6 vs 10.8 %. These hold only if the flavour
thresholds *share* the sea among flavours (Σ_f 𝒫_qg^f = 4), as the papers describe; taking the threshold
literally (n_f^eff = Σ𝒫) gives glue 42.3 %, sea 9.5 %. The sea *flavour split* (s, c; d̄ − ū) is not
reproduced exactly by any reading tried (thresholds at δ_q or at M_q = ζ_H + δ_q), and the Pauli term as
printed (g_d = +0.34 with √3(1 − 2x)) moves momentum *out of* d + d̄; the sign is chosen to give d̄ > ū as
stated. To be confirmed with the authors; valence results do not depend on any of this.

x-dependence: six moments do not fix q(x). The band in pdf_valence.png is the envelope of positive fits
(N x^a(1−x)^b; × exp(cx + dx²) with b free or b = 3) with χ²/dof ≤ 2; it is reliable for
0.15 ≲ x ≲ 0.7 only and understates the end-point uncertainty. The ζ₂ curves are refits of evolved
moments, not x-space DGLAP, so their small-x behaviour is not meaningful. Fitted hadron-scale peak:
x u_V ≈ 1.95 at x ≈ 0.39, x d_V ≈ 0.92 at x ≈ 0.37.

### Higher moments (scripts/12; results/pdf_moments_highm.json, pdf_highm_raw.json)

Converged numerics: amplitude nr = 32, ncheb = 8; one-loop (64,32,24); two-loop (16,14,10)/(16,14,10,10)
(the old (12,6,6) two-loop grid was **not** converged beyond m ~ 3: diagram 4 at m = 6 changes by 4x;
totals at m <= 6 move by <= 1 %, the zeta_3 comparison is unaffected). Each insertion carries the
regulator (1 + λp²)^(−m) (Bednar, Cloët, Tandy, arXiv:1803.03656), λ ∈ [0.005, 0.05] extrapolated to 0
with the basis {1, λ, λ log λ, λ²}: polynomial-only extrapolation is biased low (λ log λ from the UV),
this basis reproduces diquark-DA moments known at λ = 0 to 0.2–2 %.

| m | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ⟨x^m⟩_u | 1.99992 | 0.68726 | 0.27398 | 0.12146 | 0.05823 | 0.02971 | 0.01595 | 0.00896 | 0.00528 | 0.00320 | 0.00215 |
| ⟨x^m⟩_d | 0.99996 | 0.31271 | 0.11565 | 0.04838 | 0.02191 | 0.01058 | 0.00538 | 0.00287 | 0.00154 | 0.00079 | — |

Momentum sum 0.99996. Quoted errors (λ-extrapolation spread) are ≤ 1 % to m = 8; with the ncheb 6 → 8
and DA-reconstruction sensitivity a realistic error at m = 9–10 is 3–5 %.

Where it stops: the scalar-diquark pieces (diagram 1 with a scalar bystander, diagram 2 scalar) are clean
to m ~ 14 even without the regulator; everything with an axial-diquark propagator and the two-loop terms
become erratic beyond m ~ 10 for every λ. The axial-diquark propagator's K_μK_ν/m² term does not fall
off in the UV, so those integrands converge only marginally. The axial-diquark LF distribution ⟨y^m⟩_D has
negative moments for m ≥ 8 (stable under all grid changes: a model property, not noise).

Large x: (1 − x)^β with β = 3 fits m ≤ 9–10 for u and d (χ²/dof ≈ 1.1–1.2); β = 4, 5 are disfavoured;
β free is not determined. The per-quark ratio of d to u moments falls from 0.91 (m = 1) to 0.49 (m = 9)
without levelling off, so d/u(x → 1) is not yet determined. ζ₃ = 3.097 GeV valence moments with these
numbers: u 30.28, 7.61, 2.43 %; d 13.78, 3.21, 0.97 % (realistic CSM [21]: 30.4(1.3), 7.7(5), 2.5(2);
13.8(6), 3.2(2), 0.9(1)).

## Helicity-dependent DFs (scripts/13; qdq/pdf_pol.py; results/pdf_helicity_moments.json)

Axial current at Q = 0 with the vertices of Chen, Fischer, Roberts, Segovia (arXiv:2103.02054): quark
vertex γ₅[γ_μA(k²) + 2γ·k k_μA′(k²)] (the pion-pole term ∝ Q_μ and the seagulls vanish at Q = 0),
1⁺1⁺ vertex c_aa ε_{μαβν}K_ν, 0⁺↔1⁺ vertex c₁δ_μβ + c₂K_μK_β/K², no scalar-diquark coupling (parity;
checked: exactly 0). Spin projection on Λ₊γ₅γ·nΛ₊, normalised to the baryon-number current.
Flavour factors are those of the vector current with the quark-q projector (= 2103.02054 App. C).

**Diquark couplings.** Computed from the quark loop inside the diquark with our amplitudes and
conventions (axial_triangle; full 4D angular grid — the azimuth is not trivial for a vertex with free
indices; the same loop with the photon vertex gives the on-shell scalar-diquark charge 0.981 =
(7.47/7.541)², i.e. the normalisation is right). They fall steeply off-shell (c_aa = −1.11i on-shell,
−0.18i at K² = 0, sign change near K² ≈ 0.7 GeV²), so they are tabulated in the complex K² plane
(results/axial_couplings_table.json, 84 points, ~1.8 h) and used at each diquark's virtuality — i.e.
diagrams 2, 3 with a resolved diquark. With constant on-shell couplings g_A = 1.65 (diagrams 2, 3 ~3×
too large). 2103.02054 uses constants fitted to resolved nucleon-level diagrams (κ_ax^aa = 0.73,
κ_ax^sa = 0.75); with those values our diagrams 2, 3 are still 2× theirs — unexplained factor 2.

Results (converged grids as scripts/12, λ-extrapolated):

| | ours | Chen et al. 2103.02054 | Cheng et al. 2023 (realistic) |
|---|---|---|---|
| g_A | 1.157 | 1.25(3) | 1.25 |
| Δu/g_A, Δd/g_A | 0.777, −0.223 | g_A^d/g_A^u = −0.16(2) | 0.762, −0.238(6) |
| a_I = Δu + Δd | 0.553 g_A | — | 0.52(1) g_A |
| g_A fractions: 1(S), 1(A), 2, 3, 4 | 0.77, 0.071, 0.015, 0.075, 0.067 | 0.71, 0.064, 0.025, 0.13, 0.072 | — |

| m | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| ⟨x^m⟩_Δu(ζ_H) | 0.8983 | 0.3360 | 0.1345 | 0.0579 | 0.0264 | 0.0128 | 0.0066 | 0.0036 | 0.0021 | 0.0012 |
| ⟨x^m⟩_Δd(ζ_H) | −0.2585 | −0.0797 | −0.0285 | −0.0110 | −0.0045 | −0.0020 | −0.0010 | −0.0005 | −0.0003 | −0.0002 |

At ζ₂ = 2 GeV (valence non-singlet evolution, LO ΔP_qq = P_qq, axial charges invariant):
⟨x⟩_{Δu−Δd} = 0.198 (NNPDFpol1.1 0.195(14), DSSV08 0.203(9), JAM17 0.241(26); PNDME lattice
0.213(15)(22)); ⟨x²⟩_{Δu−Δd} = 0.051 (NNPDFpol 0.059–0.067, Taggi et al. lattice 0.094(15); valence only),
i.e. the same pattern as unpolarised: ⟨x⟩ on target, ⟨x²⟩ 15–20 % below fits. g_A is 9 % below
experiment (1.2754) and 7 % below 2103.02054, consistent with the 3–5 % spin-flip deficit of J, S, μ.

## Gluon helicity and proton spin (scripts/14; qdq/spin.py; results/proton_spin.json)

AO scheme as in EPJC 84:739 Sect. 5 and Cheng et al. PLB 844 (2023) 138074 (note ζ_C² = 3 GeV², i.e.
ζ_C = √3 GeV): at ζ_H all spin is carried by the dressed valence quarks, ΔG = ℓ_g = 0,
ℓ_q = ½ − ½a_I; LO evolution: a_I fixed, dΔG/dt = (α/2π)[2a_I + (β₀/2)ΔG] (n_f = 4), J_q, J_g evolve
like momentum fractions (J_g = ½⟨x⟩_g); measured quark helicity a_I^E = a_I − n_f (α/2π) ΔG.
Validation: with Cheng et al.'s input a₀ = 0.65 we get L_q(ζ_C) = −0.027 (theirs −0.027), ΔG = 1.29–1.34
(1.23), a₀^E = 0.33–0.34 (0.35(2)); with the EPJC SCI input ℓ_q = 0.071 (0.071), ΔG = 0.90–0.93 (0.91),
ℓ_g = −0.70…−0.73 (−0.71). (EPJC's a_I^E = 0.28 needs α at 3 GeV rather than √3 GeV.)
Ranges = ζ_H 0.3266 | 0.331.

Ours: a_I = Δu + Δd = 0.640 (0.553 g_A) at ζ_H.

| | a_I/2 | ℓ_q | ΔG | ℓ_g | a_I^E |
|---|---|---|---|---|---|
| ζ_C = √3 GeV | 0.320 | −0.023 | 1.27–1.32 | −1.07…−1.11 | 0.323–0.334 |
| 2 GeV | 0.320 | −0.027 | 1.39–1.43 | −1.18…−1.23 | 0.319–0.329 |
| √10 GeV | 0.320 | −0.038 | 1.75–1.81 | −1.53…−1.59 | 0.307–0.317 |

COMPASS a₀(ζ² = 3 GeV²) = 0.32(7): agreement. ΔG is large and positive, as in Cheng et al.; the χQCD
lattice glue spin S_G = 0.251(47)(16) (μ² = 10 GeV², MS-bar, Coulomb gauge) is much smaller — a known
tension of AO-scheme first moments (they include the small-x glue, which AO evolution makes larger than
fits), and the definitions/schemes differ.

Quark OAM by flavour at ζ_H: light-front (Wandzura–Wilczek, EPJC Eq. 46, ℓ_f = ½(⟨x⟩_f − Δf)):
ℓ_u = −0.106, ℓ_d = 0.286; Ji (our GFFs, L_q = J_q − ½Δq): L_u = 0.112, L_d = 0.047 (sum 0.159 vs 0.180:
the J(0) = 0.479 deficit). The two differ by B_q(0)/2 with B_u = 0.435, B_d = −0.477 from our GFFs: the
WW flavour split assumes B_q = 0 flavour by flavour, which this framework does not support.

## Proton leading-twist DA: f_N and first moments (scripts/16; qdq/nucleon_da.py; results/nucleon_da.json)

Definition as Mezrag, Segovia, Chang, Roberts (arXiv:1711.09101) Eq. 1: projection of the three-quark
Bethe–Salpeter wave function ψ₁ + ψ₂ + ψ₃ (cyclic labels; diquark loops resolved) on
ε(u₊ᵀ C† n̸ u₋) n̸ d₊.  The diquark loop is reduced analytically to σ-product integrals.  Checks: the
result is ∝ n̸H₊Λ₊ to 1e-16 (azimuth-averaged); the helicity-forbidden scalar part of ψ₂ is exactly 0;
the scalar-diquark-only shape has ⟨x₂⟩ = ⟨x₃⟩.

UV: the scalar-diquark part converges (f_N^{0⁺} = 1.13×10⁻³ GeV², shape ⟨x⟩ = 0.490, 0.255, 0.255; at
2 GeV 0.430, 0.288, 0.282 vs Mezrag et al.'s scalar-only 0.412, 0.295, 0.293).  The transverse axial
part (ψ₂: u↑d↑) converges too.  The longitudinal axial-diquark part (ψ₃ {uu}, ψ₁ {ud}; ratio −½ fixed
by flavour) diverges logarithmically in the diquark loop: the δ_μν coupling of Γ^{1⁺}_μ ~ γ_μ F(k²/ω²)
to the twist-3 operator, with F ~ 1/k² and no log suppression.  The axial-diquark DA of
qdq/diquark_df.py contracts the diquark index with n_ν and does not see this piece.  Results vs the cutoff
Λ on the diquark-loop momentum (outer loop |l| < 15 GeV; 8–30 GeV moves ⟨x₁⟩, ⟨x₃⟩ by ∓0.005 at
Λ = 5; inner radial grid 1 % on f_N):

| Λ [GeV] | f_N(ζ_H) [10⁻³ GeV²] | ⟨x₁⟩, ⟨x₂⟩, ⟨x₃⟩ (ζ_H) | 0⁺ share | at 2 GeV: f_N, ⟨x₁⟩, ⟨x₂⟩, ⟨x₃⟩ |
|---|---|---|---|---|
| 1 | 1.48 | 0.385, 0.319, 0.296 | 0.60 | 1.29, 0.365, 0.325, 0.310 |
| 2 | 2.17 | 0.362, 0.319, 0.319 | 0.51 | 1.89, 0.351, 0.325, 0.324 |
| 5 | 2.75 | 0.344, 0.319, 0.337 | 0.41 | 2.39, 0.340, 0.325, 0.335 |
| 20 | 3.47 | 0.330, 0.319, 0.351 | 0.33 | 3.02, 0.331, 0.325, 0.344 |
| 50 | 4.09 | 0.322, 0.319, 0.359 | 0.28 | 3.56, 0.326, 0.325, 0.349 |

LO evolution with the AO charge (f_N: γ = 2/3; ⟨x₁⟩ − ⟨x₃⟩: 20/9; ⟨x₁⟩ − 2⟨x₂⟩ + ⟨x₃⟩: 8/3, relative
to f_N).  RQCD 2019 (arXiv:1903.12590) at 2 GeV: f_N = 3.54(6), ⟨x⟩ = 0.396, 0.311, 0.293 (from φ₁₀,
φ₁₁); Mezrag et al. 2018: 3.78(14), 0.379, 0.302, 0.319.  For any Λ our DA is less skewed than the
lattice one, and f_N reaches the lattice value only at Λ ≈ 50 GeV, where the shape is almost
symmetric.  The divergence needs a physics decision (cutoff as a model parameter; unit-normalised diquark
DAs as in Mezrag et al.; a QCD-like log suppression of the axial-diquark amplitude) before f_N means
anything.

## Λ and Σ⁰: flavour-resolved Faddeev equation (scripts/17; qdq/baryon_faddeev.py; results/hyperon_masses.json)

Channel groups = diquark type × diquark flavour (ll, ls, ss) × bystander flavour, isospin partners sharing
one Dirac amplitude: Λ = [ud]s, ([us]d − [ds]u)/√2, ({us}d − {ds}u)/√2; Σ⁰ = ([us]d + [ds]u)/√2, {ud}s,
({us}d + {ds}u)/√2.  Flavour factors c^{(e)}_{gg'} = Σ F^g_{kj,i} F^{g'}_{ij,k}, split by the exchanged
quark's flavour e (which fixes S_e^T in the kernel).  Strange quark: Chen, Krein, Roberts, Schmidt,
arXiv:1901.04305 App. A (λ = 0.566 GeV, m̄_s = 0.210; M^E_s = 0.495 GeV).  Equal momentum partition
inside every diquark; η = 1/3.

Checks: proton flavour factors and couplings reproduce qdq/faddeev.py and qdq/diquark.py; proton
λ(1.18) on (8,4,4) = 1.0154017190; in the SU(3) limit Λ and Σ⁰ have exactly the nucleon eigenvalue.

| diquark masses [GeV] | 2\|g\| (0⁺ll, 1⁺ll, 0⁺ls, 1⁺ls, 1⁺ss) | M_N | M_Λ | M_Σ⁰ |
|---|---|---|---|---|
| C: 0.80, 0.90, 0.95, 1.05, 1.20 (1901.04305) | 14.75, 12.73, 9.45, 7.62, 3.72 | 1.195 | 1.380 | 1.405 |
| 1901.04305 Eq. (7), Table I | 14.75, 12.73, 9.45, 7.62, 3.72 | 1.19(13) | 1.37(14) | 1.41(14) |
| A: 0.79, 0.89, 0.94, 1.04, 1.19 (2611.19819 + 0.15 per s) | 15.08, 13.02, 9.69, 7.84, 3.87 | 1.174 | 1.360 | 1.385 |

Couplings from the canonical normalisation (all five agree with 1901.04305 to the digits printed);
dressed-quark-core masses, grid-converged to 1e-4 GeV ((nr, ncheb) = (16, 6) vs (24, 8)).

## Λ and Σ⁰ valence DFs: Mellin moments (scripts/18; qdq/baryon_pdf.py; results/hyperon_pdf_moments.json)

Same construction as the proton (qdq/pdf.py: forward number current of flavour q, light-front weights,
normalisation from dλ/dM), now over the channel groups: each group has its own bystander propagator,
diquark mass, width and coupling; the exchanged quark is light or strange.  Flavour × charge factors from
the group tensors F^g_{ab,c} with a diagonal charge t^q: diagram 1 Σ F t_c F, diagram 2 Σ F (t_a + t_b) F,
diagrams 4–6 Σ F^g_{kj,i} [t_j | t_i | t_k] F^{g'}_{ij,k} (exchanged quark | outgoing bystander, seagull 5 |
incoming bystander, seagull 6).  Diagram 3 (0⁺ ↔ 1⁺) vanishes at Q = 0.  Setup A, all couplings from the
canonical normalisation; grids as scripts/09 (moments reliable to m ≈ 6).

Quark-in-diquark DFs: q(z) ∝ φ² as for the proton, with the DA of the unequal-flavour diquark computed
from the same amplitude (S_l(k₊) Γ C† S_s(k₋), equal momentum partition; qdq/diquark_df.da_moments_flavour)
and fitted with Gegenbauer terms of both parities to ⟨ξ¹…ξ⁶⟩ (fit_da_gen; for ll diquarks identical to
the proton fit).  Light quark ∝ φ(z)², strange ∝ φ(1−z)²:

| diquark | ⟨ξ⟩ (ξ = 2x_l − 1) | ⟨z⟩_l, ⟨z⟩_s of φ² |
|---|---|---|
| [us] 0⁺, 0.94 GeV | +0.081 | 0.562, 0.438 |
| {us} 1⁺, 1.04 GeV | −0.0025 | 0.499, 0.501 |

In the [us] the *light* quark carries more momentum: the 0⁺ trace is ∝ x σ_V^l σ_S^s + (1−x) σ_S^l σ_V^s,
so the strange quark's larger σ_S weights the light fraction (the trend grows if the s quark is made
heavier).  The same E-term structure appears in the SCI [ls] DF of 2505.10663 Eq. (A.7a).  Checked: swapping
the legs gives ξ → −ξ exactly.  A symmetric [us] DF instead would raise ⟨x⟩_s by 0.004 (Λ), 0.010 (Σ⁰).

Checks: proton from BaryonFaddeev = NucleonPDFMoments to 6e-12 (all m); SU(3) limit (m_s = m_l,
degenerate diquarks): M_Λ = M_Σ = M_N and the octet relations Σ⁰: u = u_p/2, s = d_p;
Λ: u = (u_p + 4d_p)/6, s = (2u_p − d_p)/3 hold for every diagram and moment to 1e-12
(results/hyperon_pdf_su3_check.json); valence numbers 1.001–1.002 each, momentum sums 1.003–1.004 (grid);
kmax 10 vs 15 GeV ≤ 2e-4; imaginary parts ≤ 3e-12.

| | M [GeV] | ⟨x⟩_l (ζ_H) | ⟨x⟩_s (ζ_H) | SCI (2505.10663 Eq. 59) l, s | ⟨x⟩, ⟨x²⟩, ⟨x³⟩ at 2 GeV [%] | SCI Table 4 |
|---|---|---|---|---|---|---|
| p (u, d) | 1.174 | 0.689, 0.314 | — | 0.686, 0.314 | u 32.9, 8.66, 2.87; d 15.0, 3.66, 1.14 | u 32.7, 9.47, 3.58; d 15.0, 4.15, 1.54 |
| Λ | 1.360 | 0.314 | 0.374 | 0.303, 0.394 | u+d 30.0, 7.22, 2.19; s 17.9, 4.90, 1.64 | u+d 28.9, 7.60, 2.66; s 18.8, 5.83, 2.29 |
| Σ⁰ | 1.385 | 0.332 | 0.340 | 0.328, 0.345 | u+d 31.7, 7.88, 2.44; s 16.2, 4.17, 1.34 | u+d 31.3, 8.46, 2.99; s 16.4, 4.70, 1.74 |

(AO evolution as for the proton, valence factors 0.477, 0.314, 0.234 for n = 1, 2, 3; results/hyperon_pdf_evolved.json.)
Momentum fractions agree with the SCI within 0.02; the s quark in Λ carries a little less than in the
SCI (0.374 vs 0.394), and, as for the proton, the higher moments are smaller (softer large-x DFs than the
SCI's (1−x)¹).  ⟨x^m⟩_s/⟨x^m⟩_l rises with m: 1.19 → 1.99 (m = 1 → 6) in Λ, 1.02 → 1.22 in Σ⁰, so s/l
exceeds 1 at large x in both, much more so in Λ.  Diagram breakdown at m = 1 (Λ, s): bystander 0.264,
in diquark 0.070, exchange 0.016, seagulls 2 × 0.013; the exchange and seagull diagrams (absent in the
SCI current) carry 11 % (s) and 19 % (u) of ⟨x⟩ in Λ.

### Sensitivity of the SU(3) breaking to the inputs (scripts/18 --scan; results/hyperon_pdf_scan.json)

Scale-independent ratios (valence evolution is flavour blind): Σ⁺ from Σ⁰ by isospin (u = 2u_Σ⁰, s = s_Σ⁰).

| setup | M_N, M_Λ, M_Σ⁰ [GeV] | ⟨x⟩_l, ⟨x⟩_s Λ | ⟨x⟩_l, ⟨x⟩_s Σ⁰ | s/l (Λ) | u_Σ⁺/u_p | s_Σ⁺/d_p |
|---|---|---|---|---|---|---|
| A10: +0.10 GeV per s | 1.174, 1.297, 1.317 | 0.318, 0.368 | 0.338, 0.329 | 1.156 | 0.980 | 1.048 |
| A: +0.15 | 1.174, 1.360, 1.385 | 0.314, 0.374 | 0.332, 0.340 | 1.191 | 0.964 | 1.083 |
| A20: +0.20 | 1.174, 1.420, 1.450 | 0.312, 0.380 | 0.328, 0.349 | 1.218 | 0.950 | 1.113 |
| C (1901.04305 masses) | 1.195, 1.380, 1.405 | 0.315, 0.373 | 0.332, 0.340 | 1.184 | 0.965 | 1.081 |
| A, m̄_s × 1.5 (M^E_s 0.495 → 0.552) | 1.174, 1.356, 1.384 | 0.314, 0.375 | 0.333, 0.339 | 1.193 | 0.965 | 1.081 |
| lattice | | | | 1.35(8) quenched (hep-lat/0208017) | 0.963(9) | 1.082(18) |

(lattice ratios: 1012.0215 Table I, N_f = 2+1 at m_π = 334, m_K = 463 MeV, where m_K² − m_π² is 45 % of its
physical value; their points are linear in it, and a naive linear extrapolation — ours, not theirs — gives
≈ 0.92 and ≈ 1.18 at the physical point.)

The breaking is set by the diquark mass spacing (roughly linear in it); the dressed s-quark mass alone, at
fixed diquark masses, changes nothing visible, and setup C equals A.  Matching the lattice trend would need
≈ 0.3 GeV per s quark (≈ 0.4 for the quenched Λ ratio), which would push M_Λ − M_N to ≈ 0.35 GeV; the
spacing 0.15 GeV gives M_Λ − M_N = 0.186 GeV (empirical 0.177).  M_Σ − M_Λ is 0.020–0.030 GeV in all setups
(empirical 0.077; 1901.04305: 0.04; SCI: 0.05).
