# 核子 quark + diquark 程序（Fortran）

按介子 DSE/BSE/形状因子程序（pi-ff-chey）的结构组织，内容与 `../qdq/`（Python 参考实现）一一对应，
所有阶段都已和 Python 逐位核对（见下）。

## 编译与运行

```bash
mkdir build && cd build && cmake .. && make          # 串行版 -> debug/nucleon-qdq
cmake .. -DUSE_MPI=ON && make                         # MPI 版（mpif90, -DMPI）
cd .. && ./debug/nucleon-qdq                          # 读 Configures.nlt
OPENBLAS_NUM_THREADS=1 mpirun -np 4 ./debug/nucleon-qdq   # MPI 并行（外点分配给各进程）
```
需要 gfortran、LAPACK/BLAS（CMake 默认找 OpenBLAS，可在 CMakeLists.txt 里去掉 `BLA_VENDOR`）。

## 运行模式（`Configures.nlt` 的 `&run_list`）

| mode | 内容 | 输出 |
|---|---|---|
| `check` | 第 1–4 步检验：传播子、M^E、diquark 归一化、味道系数、固定网格上的本征值 λ(M) | 屏幕；`outdata/diquark.txt` |
| `faddeev` | 解 Faddeev 方程，求 M_N 和振幅 | `outdata/faddeev/amplitude.txt` |
| `em` | 核子电磁形状因子（p, n），六个图，逐图输出 | `outdata/current/em_ff.txt` |
| `gff` | 质子引力形状因子 A, J, D, S, c̄，逐图输出 | `outdata/current/grav_ff.txt` |
| `pdf` | 非极化价夸克 PDF 的 Mellin 矩：各图、各 λ、m = 0..mmax 的加权前向流 | `outdata/pdf/pdf_raw.txt` |
| `axial_table` | 双夸克轴矢耦合 c(K²)（双夸克内的夸克圈，复 K² 平面网格），`helicity` 要用 | `outdata/pdf/axial_couplings.txt` |
| `helicity` | 螺旋度 PDF 的矩（轴矢流，Q = 0），同上 | `outdata/pdf/helicity_raw.txt` |
| `da` | 0⁺、1⁺ 双夸克的领头扭度 DA 的矩（双夸克内夸克分布 φ² 用） | `outdata/pdf/da_raw.txt` |

`em`、`gff` 开头会先打印几项自洽检验：Ψ̄G₀KG₀Ψ = Ψ̄G₀Ψ、重子数 Ward 恒等式、归一化。

**PDF 的流程**（参数在 `&pdf_para_list`）：先跑一次 `axial_table`（只在改了夸克/双夸克参数时重跑），
再跑 `pdf`、`helicity`，最后在仓库根目录跑
```bash
python scripts/15_fortran_post.py fortran/outdata/pdf
```
它读 Fortran 的输出，做 λ → 0 外推、乘双夸克内夸克分布的矩（φ²、φ₀φ₁）、AO 演化，打印结果并写
`outdata/pdf/post.json`。Fortran 只做积分（重的部分），外推、拟合、演化这些轻的步骤留在 Python。
- 快速检验：默认的粗网格和 λ = 0, 0.01, 0.03（λ = 0 的矩可靠到 m ≈ 6），约 1 分钟。
- 收敛的结果（README 第 7–9 节的数）：配置在 `runs/fine/Configures.nlt`（Faddeev `nr = 32, nz = 10,
  ncheb = 8`，`nko = 16, 14, 10`，`nli = 16, 14, 10, 10`，`kmax = 30`，9 个 λ，`mmax = 16`），在该目录运行
  `./run.sh`，再 `python scripts/15_fortran_post.py fortran/runs/fine/outdata/pdf`；矩可靠到 m ≈ 9–10。

## 目录结构（对照介子程序）

```
Configures.nlt                 所有参数（namelist），同介子程序
sources/_main.f90              主程序：读 run_list，选择模式
sources/calc.f90               驱动：run_checks / run_faddeev / run_form_factors
sources/_libraries/            通用库
    constant.f90   pi, Im, Zero, One, hbarc, dot4（欧氏点积，不取共轭）
    dirac.f90      γ 矩阵、C、γ5、σ_{μν}、ε、slash、tr2
    quadrules.f90  Gauss-Legendre、径向映射、Chebyshev 第二类（与 Python 完全相同）
    special.f90    F(x) = (1-e^{-x})/x
    spline.f90     复数三次样条（not-a-knot，= scipy CubicSpline）
    linearsys.f90  zgeev（实部最大本征值）、矩阵求逆
    mpich2.f90     MPI 封装（MPI_initialize / MPI_distribute / MPI_reducesum），无 -DMPI 时串行
    iostream.f90   x_mkdir
sources/zero_quark/            夸克传播子（代数参数化，2611.19819 App. A.1）
sources/zero_diquark/          diquark 振幅、传播子、正则归一化 → g₀₊, g₁₊
sources/zero_faddeev/          Faddeev 方程
    faddeev_elems.f90   参数、味道系数 c_XY、正能基矢、对偶基
    faddeev_solve.f90   核矩阵、本征值、割线法求 M_N、Chebyshev 振幅、任意动量处的振幅 Ψ 与 Ψ̄
sources/zero_normalization/    由 dλ/dM 得到的正则归一化（不需要双圈图）
sources/zero_probe/            光子 / 引力子探针：顶点、味道系数、seagull、投影算符
sources/zero_current1/         单圈图 1–3（以及积分网格、Breit 系运动学）
sources/zero_current2/         双圈图 4–6（规范化 Faddeev 核；每个外点一次 zgemm）
sources/zero_pdf/              PDF 的矩
    pdf_elems.f90   参数、光锥 n、权重 [x/(1+λp²)]^m、前向轴矢夸克顶点、自旋投影 Λ₊γ₅γ·nΛ₊
    axial_elems.f90 双夸克轴矢耦合：夸克圈（完整 4 维角度网格）、复 K² 表、双三次样条插值
    pdf_solve.f90   单圈图 1–3、双圈图 4–6 的加权迹（数流 / 轴矢流），双夸克 DA 的矩
```
每个阶段的写法同介子程序：`*_elems`（参数类型带默认值 + 元素）、`*_utils`（namelist 读入）、
`*_solve`（方程类型，带 `init / kernel / solve / output / free` 等类型绑定过程）。

## 和 Python 的核对结果

| 阶段 | 量 | Fortran = Python |
|---|---|---|
| 1 | σ_V, σ_S（实与复 p²），M^E = 0.3275076942 | 逐位 |
| 2 | g₀₊² = 56.868825，g₁₊² = −42.400818 | 逐位 |
| 3–4 | λ(1.10), λ(1.18) 在 (8,4,4)、(16,8,8) 网格，s 与 sa 共 8 个值，如 λ(1.18) = 1.0154017190 | 10 位 |
| 5 | M_N = 1.17868 GeV（N_cheb = 6），dλ/dM = 2.916520 | 逐位 |
| 6 | 重子数 Ward 恒等式 0.0058007i，归一化 0.0072269i，F₁ 逐图 | 逐位 |
| 7 | Ψ̄G₀KG₀Ψ = 0.0024768；F₁ᵖ(0) = 1.00024，F₁ⁿ(0) = 0.00053，x = 0.02 的 G_E, G_M | 逐位 |
| 8 | A(0) = 1.00375（逐图），x = 0.02 的 A, J, D, S, c̄（逐图） | 逐位 |
| 9 | 非极化矩：u, d 数 2.00103 / 1.00131，各图（1, D_s, D_a, 4, 5, 6）、各 λ、m ≤ 6 | 相对差 ≤ 2×10⁻⁷ |
| 10 | 双夸克轴矢耦合表（84 个复 K² 点 × 5 个耦合） | 绝对差 ≤ 4×10⁻⁸ |
| 11 | 螺旋度矩（Python 的耦合表）：g_A = 1.18529，各图 m ≤ 4 | 逐位（5 位小数） |
| 12 | 双夸克 DA 的 ⟨ξ²⟩, ⟨ξ⁴⟩, ⟨ξ⁶⟩ = 0.19013, 0.07547, 0.03925（0⁺），0.10904, 0.03067, 0.01237（1⁺） | 逐位 |
| 13 | 收敛网格（`runs/fine`）、λ 外推后的 ⟨x^m⟩_u,d（m ≤ 10）与 ⟨x^m⟩_Δu,Δd（m ≤ 9），螺旋度用 Fortran 自己的耦合表 | 差 ≤ 7×10⁻⁹（外推误差 ~10⁻⁵） |

速度（4 核容器，粗双圈网格）：电磁两个 Q² 点串行 1.8 分钟、MPI 4 进程 36 秒；引力两个 Q² 点约 6 分钟（串行）；
`pdf`、`helicity` 各约 1 分钟；`axial_table` 11 分钟（Python 1.8 小时）。
收敛网格（`runs/fine/run.sh`，MPI 4 进程）：`pdf` 10 分钟、`helicity` 6 分钟（Python 约 80、60 分钟）。

## 学习建议

1. 先跑 `check` 模式，读 `_libraries/dirac.f90`、`zero_quark`、`zero_diquark`，弄清约定。
2. **自己重写 `zero_faddeev/faddeev_solve.f90` 的 `kernel` 子程序**（这是整个框架的核心），
   用 `check` 模式核对 λ(1.18) = 1.0154017190。写之前先不要看现成的实现。
3. 读 `zero_probe` 和 `zero_current1`，对照 `../docs/fortran_guide.md` 第 6–8 步的公式和坑。
4. PDF：读 `zero_pdf/pdf_solve.f90` 的 `pdf_one_loop`，对照 `../qdq/pdf.py` 开头的公式；
   跑 `pdf`（粗网格）验证 u 数 = 2、d 数 = 1、⟨x⟩_u + ⟨x⟩_d = 1。
5. 小练习：在 `exchange_flavour` 里把电荷换成 τ^u、τ^d，验证 u 数 = 2、d 数 = 1；
   把 `probe_type%r_sc`、`r_av` 设为 0，验证引力 c̄(Q²) 处处为 0。

已知问题（与 Python 版本相同，因为是同一套实现）：J(0) ≈ 0.479，比要求的 ½ 低约 4%，
S(0)、μ_p、μ_n 也低 3–5%；见 `../README.md` 的引力形状因子一节。
