!Nucleon quark+diquark Faddeev equation (= qdq/faddeev.py)
!
!   psi_X(k) = -4 sum_Y c_XY g_X g_Y int d^4l/(2pi)^4
!              Gamma^Y(k_q - l_qq/2) S^T(l_qq - k_q) Gammabar^X(l_q - k_qq/2) S(l_q) Delta^Y(l_qq) psi_Y(l)
!   l_q = l + P/3, l_qq = -l + 2P/3 (same for k);  -4 = 2 (Psi_3 <-> Psi_1,2) x (-2) (colour)
!   (Alkofer et al., nucl-th/0412046, Eqs. 8-13)
!
!Rest frame P = (0,0,0,iM).  Amplitude f_c(l^2,z) = sum_n f_{c,n}(|l|) U_n(z), z = l_4/|l| = -i l.Phat/|l|.
!External points k = |k|(0,0,sqrt(1-z^2),z); integration l = |l|(0, s_z s_y, s_z y, z), azimuth -> 2 pi.
!Unknown vector ordering: idx = m + nm*(i-1) + nm*nr*(c-1)  (moment m fastest, then radial i, channel c).
! ====================================================================================================
! 【注释版说明】
! 本文件与 sources/zero_faddeev/faddeev_solve.f90 的代码逐字相同，只多了以 "!" 开头的中文注释，
! 供阅读用；它不在 sources/ 下，不参与编译。改代码请改 sources/ 里的原文件。
! Python 对照：qdq/faddeev.py 的 NucleonFaddeev 类（kernel_matrix、eigen、solve_mass、
! scalar_functions）。
!
! ----------------------------------------------------------------------------------------------------
! 一、这个模块做什么
!   解核子的夸克 + 双夸克 Faddeev 方程（两体 BSE 的形式）：
!      Ψ_X(k;P) = -4 Σ_Y c_XY g_X g_Y ∫ d^4l/(2π)^4
!                 Γ^Y(k_q - l_qq/2) S^T(l_qq - k_q) Γbar^X(l_q - k_qq/2) S(l_q) Δ^Y(l_qq) Ψ_Y(l;P)
!   X, Y = s（标量双夸克 0+）或 a（轴矢双夸克 1+）；轴矢道还带 Lorentz 指标，这里省略。
!   做法与介子 BSE 相同：把右边离散成矩阵 K(M)，求最大本征值 λ(M)，调 M 使 λ(M) = 1。
!
! 二、记号（全部欧氏度规，a·b = Σ_{i=1..4} a_i b_i）
!   P = (0,0,0,iM)       核子总动量（静止系），P^2 = -M^2
!   l, k                 夸克-双夸克相对动量：l 是积分（内点），k 是方程左边的外点
!   l_q  = l + P/3       内点旁观夸克的动量（η = 1/3）
!   l_qq = -l + 2P/3     内点（入射）双夸克的动量
!   k_q, k_qq            外点同理
!   q = l_qq - k_q       被交换的夸克的动量（以 S^T 出现）
!   rin  = k_q - l_qq/2  入射双夸克 Y 拆开时的相对动量（Γ^Y 的自变量）
!   rout = l_q - k_qq/2  出射双夸克 X 形成时的相对动量（Γbar^X 的自变量）
!   z = l_4/|l|          四维角的余弦（l·Phat = i z |l|）
!   y                    外点 k 与内点 l 的三维部分之间夹角的余弦
!
! 三、未知数怎么存
!   振幅 = 8 个标量函数 f_c(|l|, z) 乘协变基矢：c = 1,2 是标量道 S^1、S^2；c = 3..8 是轴矢道
!   γ5 A^1..A^6（2611.19819 Eq. A15）。每个 f_c 按第二类 Chebyshev 多项式展开：
!      f_c(|l|, z) = Σ_{n=0}^{nm-1} f_{c,n}(|l|) U_n(z)
!   未知数 = f_{c,n}(r_i)，一维编号 idx = m + nm*(i-1) + nm*nr*(c-1)（m 最快，其次径向 i，最后道 c）。
!   只开一种双夸克时（channels = 's' 或 'a'），nch = 2 或 6，未知数只含那几个道。
!
! 四、调用顺序
!   init(para)   建网格、外点、内点，以及外点的对偶基、内点的基矢
!   kernel(M)    组装矩阵 K(M)                         ← 本文件的核心
!   eigen(M)     对 K(M) 求实部最大的本征值 λ(M) 和本征向量
!   solve()      割线法解 λ(M) = 1，然后 store() 保存振幅
!   sfun / amplitude / conj_amplitude   在任意动量处取振幅（算流时用）
! ====================================================================================================
module faddeev_solve
    use libraries
    use quark_elems
    use diquark_elems
    use faddeev_elems
    use faddeev_utils
    implicit none

private

! ----------------------------------------------------------------------------------------------------
! 类型 faddeev_equation 的成员
!   para          输入参数（网格、耦合 g、双夸克道、割线法初值等；见 faddeev_elems.f90）
!   M             解出的核子质量
!   gs, ga        双夸克耦合 g_{0+}、g_{1+}（g_{1+} 在我们的 γ 表示下是虚数，必须保留相位）
!   c(2,2)        味因子 c_XY（1 = 0+，2 = 1+）：c_ss = 1/2，c_sa = c_as = -√3/2，c_aa = -1/2
!   nr, nz, ny    径向、z、y 的积分节点数
!   nm            每个道的 Chebyshev 矩个数（Nystrom 模式下 nm = nz）
!   nza           外点的角度节点数（Chebyshev 模式下 = ncheb；Nystrom 模式下 = nz）
!   ni            内点总数 = nr*nz*ny
!   nch, off(2)   开启的道数，以及标量、轴矢道在未知数里的起始偏移
!   r, wr         径向节点 |l| 和权重；z, wz：Chebyshev 节点和权重；y, wy：Gauss-Legendre 节点和权重
!   zext, wext    外点的 z 节点和权重
!   Pout(nm,nza)  外点角度上的函数值 -> Chebyshev 矩
!   Ein(nz,nm)    Chebyshev 矩 -> 内点 z 节点上的函数值
!   kvec          外点 k（4, nr, nza）；lvec, wl：内点 l（4, ni）和积分权重（含 2π 方位角和 1/(2π)^4）
!   Bs_l, Ba_l    内点的基矢（已右乘 Λ+）；Dts, Dta：外点的对偶基（取系数用）
!   f, fy, f2     解出的振幅矩 f(c, n, i)，以及在 log|l| 上做三次样条用的数据和二阶导数
! ----------------------------------------------------------------------------------------------------
type faddeev_equation
    type(faddeev_para),public:: para
    real(8),public:: M
    complex(8),public:: gs,ga
    real(8),public:: c(2,2)
    integer,public:: nr,nz,ny,nm,nza,ni,nch,off(2)
    logical,public:: use_s,use_a
    real(8),allocatable,public:: r(:),wr(:),z(:),wz(:),y(:),wy(:),zext(:),wext(:),Pout(:,:),Ein(:,:)
    complex(8),allocatable,public:: kvec(:,:,:),lvec(:,:),wl(:)
    complex(8),allocatable:: Bs_l(:,:,:,:),Ba_l(:,:,:,:,:),Dts(:,:,:,:,:),Dta(:,:,:,:,:,:)
    complex(8),allocatable,public:: f(:,:,:),fy(:,:),f2(:,:)   !moments f(c,n,i); spline data/2nd derivatives
    real(8),allocatable:: logr(:)
contains
    procedure,public:: init         => faddeev_equation_init
    procedure,public:: free         => faddeev_equation_free
    procedure,public:: kernel       => faddeev_equation_kernel
    procedure,public:: eigen        => faddeev_equation_eigen
    procedure,public:: solve        => faddeev_equation_solve
    procedure,public:: store        => faddeev_equation_store
    procedure,public:: sfun         => faddeev_equation_sfun
    procedure,public:: amplitude    => faddeev_equation_amplitude
    procedure,public:: conj_amplitude => faddeev_equation_conj_amplitude
    procedure,public:: output       => faddeev_equation_output
end type faddeev_equation

public faddeev_equation

contains

    ! ====================================================================================================
    ! init：准备网格和基矢。只和网格有关，与 M 无关，所以只做一次。
    ! ====================================================================================================
    subroutine faddeev_equation_init(this,para)
        class(faddeev_equation):: this
        type(faddeev_para):: para
        integer:: i,a,j,b,iy,p,n
        real(8):: sz,sy
        complex(8):: lhat(4),Phat(4),Bs(4,4,2),Ba(4,4,4,6),Lp(4,4)
        call this%free()
        this%para = para
        ! 双夸克耦合：默认取文章值的一半 g = (7.47, 6.49 i)，与核前因子 -4 配套。
        this%gs = para%gs; this%ga = para%ga
        ! 开哪些道：'s' 只有标量双夸克（2 个未知函数），'a' 只有轴矢（6 个），'sa' 两者耦合（8 个）。
        ! off(1)、off(2) 是标量、轴矢道在未知数里的起始位置。
        this%use_s = index(para%channels,'s')>0
        this%use_a = index(para%channels,'a')>0
        this%nch = 0
        if(this%use_s)then
            this%off(1) = 0; this%nch = 2
        end if
        if(this%use_a)then
            this%off(2) = this%nch; this%nch = this%nch+6
        end if
        ! 味因子 c_XY，质子味旋量 φ = (1, 0)。推导见学习笔记附录；中子给出同样的值。
        this%c = flavour_coeffs([1d0,0d0])
        ! 积分网格：
        !   径向  |l| = s·t/(1-t)，t 取 Gauss-Legendre 节点，s = para%scale（默认 0.6 GeV），
        !         权重 wr 已含 dt -> d|l| 的雅可比
        !   z     第二类 Gauss-Chebyshev：Σ wz f(z) ≈ ∫_{-1}^{1} dz √(1-z^2) f(z)（四维测度里正好有 √(1-z^2)）
        !   y     Gauss-Legendre，∫_{-1}^{1} dy
        !   最后一个方位角积出 2π（投影后被积函数只依赖标量积，与它无关）
        this%nr = para%nr; this%nz = para%nz; this%ny = para%ny
        allocate(this%r(this%nr),this%wr(this%nr),this%z(this%nz),this%wz(this%nz),this%y(this%ny),this%wy(this%ny))
        call radial(this%nr,para%scale,this%r,this%wr)
        call chebyshev_u(this%nz,this%z,this%wz)
        call gauleg(this%ny,this%y,this%wy)
        ! 外点的角度离散有两种模式：
        !   ncheb <= 0（Nystrom）：未知数就是振幅在 (|l|, z) 节点上的值，外点 z = 积分 z，Pout = Ein = 单位矩阵。
        !                         docs/reference_values.md 里的参考本征值用的是这种模式。
        !   ncheb  > 0（Chebyshev）：未知数是 Chebyshev 矩 f_{c,n}。
        !      Ein(b,n)  = U_n(z_b)：由矩重建内点 z_b 处的函数值，f(z_b) = Σ_n U_n(z_b) f_n
        !      Pout(m,a) = (2/π) wext_a U_m(zext_a)：由外点 zext_a 处的函数值求矩
        !                  （用到 ∫ √(1-z^2) U_m U_n dz = (π/2) δ_mn，外点取 ncheb 个 Chebyshev 节点时精确）
        if(para%ncheb<=0)then                                !Nystrom: external z = integration z
            this%nm = this%nz; this%nza = this%nz
            allocate(this%zext(this%nza),this%wext(this%nza),this%Pout(this%nm,this%nza),this%Ein(this%nz,this%nm))
            this%zext = this%z; this%wext = this%wz
            this%Pout = 0d0; this%Ein = 0d0
            do i=1,this%nz
                this%Pout(i,i) = 1d0; this%Ein(i,i) = 1d0
            end do
        else
            this%nm = para%ncheb; this%nza = para%ncheb
            allocate(this%zext(this%nza),this%wext(this%nza),this%Pout(this%nm,this%nza),this%Ein(this%nz,this%nm))
            call chebyshev_u(this%nza,this%zext,this%wext)
            do a=1,this%nza
                this%Pout(:,a) = 2/pi*this%wext(a)*real(cheb_u(this%nm,dcmplx(this%zext(a),0d0)))
            end do
            do b=1,this%nz
                this%Ein(b,:) = real(cheb_u(this%nm,dcmplx(this%z(b),0d0)))
            end do
        end if
        ! 外点 k 与对偶基：
        !   k = |k| (0, 0, √(1-z^2), z)：三维部分沿第 3 轴；|k| 取径向节点，z 取 nza 个外点角度节点。
        !   basis() 给出 2 + 6 个协变基矢（轴矢道含 γ5，见 faddeev_elems.f90），这里再右乘 Λ+ = (1+γ4)/2
        !   （静止系下的正能投影 (1 - iγ·P/M)/2）。
        !   基矢在迹内积下不正交，不能直接用迹取系数。duals() 先求 Gram 矩阵 G_mn = Σ Tr[B_m† B_n] 并求逆，
        !   得到“对偶基” D_m = Σ_n (G^{-1})_mn B_n†，于是任何 Dirac 矩阵 R 在 B_m 上的系数是 tr2(D_m, R)
        !   （轴矢道对 Lorentz 指标求和）。外点的对偶基与 M 无关，预先存好。
        !external points and their dual bases
        allocate(this%kvec(4,this%nr,this%nza),this%Dts(4,4,2,this%nr,this%nza),this%Dta(4,4,4,6,this%nr,this%nza))
        Phat = [Zero,Zero,Zero,Im]
        Lp = 0.5d0*(id4+gm(:,:,4))                          !Lambda_+ for P = (0,0,0,iM)
        do i=1,this%nr
            do a=1,this%nza
                this%kvec(:,i,a) = this%r(i)*[0d0,0d0,sqrt(1-this%zext(a)**2),this%zext(a)]
                lhat = this%kvec(:,i,a)/this%r(i)
                call basis(lhat,Phat,Bs,Ba)
                do n=1,2
                    Bs(:,:,n) = matmul(Bs(:,:,n),Lp)
                end do
                do n=1,6; do j=1,4
                    Ba(:,:,j,n) = matmul(Ba(:,:,j,n),Lp)
                end do; end do
                call duals(Bs,Ba,this%Dts(:,:,:,i,a),this%Dta(:,:,:,:,i,a))
            end do
        end do
        ! 内点 l（积分点）与基矢：
        !   l = |l| (0, s_z s_y, s_z y, z)，s_z = √(1-z^2)，s_y = √(1-y^2)；y 是 l 与 k 三维部分的夹角余弦。
        !   方位角已经积成 2π，所以第 1 个分量取 0 即可。
        !   权重 wl = wr·|l|^3·wz·wy·2π/(2π)^4，对应 ∫ d^4l/(2π)^4。
        !   内点存放顺序：径向 i 最慢，然后 z，y 最快（kernel 里按同样的顺序遍历）。
        !   内点的基矢 Bs_l、Ba_l 也右乘了 Λ+，用来把“矩 -> 内点处的振幅”写成 Σ f_n U_n(z) B_c。
        !integration points (r, z, y) with y fastest, and the bases there (times Lambda_+)
        this%ni = this%nr*this%nz*this%ny
        allocate(this%lvec(4,this%ni),this%wl(this%ni),this%Bs_l(4,4,2,this%ni),this%Ba_l(4,4,4,6,this%ni))
        p = 0
        do i=1,this%nr
            do b=1,this%nz
                do iy=1,this%ny
                    p = p+1
                    sz = sqrt(1-this%z(b)**2); sy = sqrt(1-this%y(iy)**2)
                    this%lvec(:,p) = this%r(i)*[0d0,sz*sy,sz*this%y(iy),this%z(b)]
                    this%wl(p) = this%wr(i)*this%r(i)**3*this%wz(b)*this%wy(iy)*2*pi/(2*pi)**4
                    lhat = this%lvec(:,p)/this%r(i)
                    call basis(lhat,Phat,this%Bs_l(:,:,:,p),this%Ba_l(:,:,:,:,p))
                    do n=1,2
                        this%Bs_l(:,:,n,p) = matmul(this%Bs_l(:,:,n,p),Lp)
                    end do
                    do n=1,6; do j=1,4
                        this%Ba_l(:,:,j,n,p) = matmul(this%Ba_l(:,:,j,n,p),Lp)
                    end do; end do
                end do
            end do
        end do
        this%M = 0d0
    end subroutine

    ! free：释放所有动态数组（init 开头会先调用，允许重复 init）。
    subroutine faddeev_equation_free(this)
        class(faddeev_equation):: this
        if(allocated(this%r)) deallocate(this%r,this%wr,this%z,this%wz,this%y,this%wy)
        if(allocated(this%zext)) deallocate(this%zext,this%wext,this%Pout,this%Ein)
        if(allocated(this%kvec)) deallocate(this%kvec,this%Dts,this%Dta)
        if(allocated(this%lvec)) deallocate(this%lvec,this%wl,this%Bs_l,this%Ba_l)
        if(allocated(this%f)) deallocate(this%f,this%fy,this%f2,this%logr)
    end subroutine

    ! ====================================================================================================
    ! kernel：组装矩阵 K(M)，使得在束缚态处 f = K(M) f（f 为未知数向量）。       ← 本文件的核心
    !
    ! 【数学】对外点 k 和出射道 X 的第 kk 个基矢，方程右边的系数是
    !    f^X_kk(k) = Σ_Y Σ_n ∫ d^4l/(2π)^4  pref · tr[ D^X_kk(k) · Γ^Y · S^T(q) · Γbar^X · S(l_q) · Δ^Y(l_qq) · B^Y_n(l) ]
    !                × f^Y_n(l)
    !    pref = -4 c_XY g_X g_Y F_Y(rin^2) F_X(rout^2) · (积分权重)
    !   其中 F 是双夸克振幅里的标量因子 F(x) = (1-e^{-x})/x（Famp），Γ、Γbar 只剩常数 Dirac 结构：
    !    Γ^{0+} = C iγ5（gam_s），Γ^{1+}_ρ = iγ_ρ C（gam_a），Γbar = C† Γ^T C（bar_s、bar_a）。
    !   轴矢道里：出射双夸克 X 的指标 μ 同时出现在 Γbar^X_μ 和外点对偶基 D_kk,μ 上；
    !            入射双夸克 Y 的指标 ρ 出现在 Γ^Y_ρ 和 Δ^Y_{ρν} B_n,ν 上。
    !
    ! 【分解】把迹里的矩阵分成三段，各自只依赖一部分变量，预先算好以省时间：
    !    tr[ (D_kk Γ^Y) · S^T(q) · (Γbar^X S(l_q) Δ^Y B_n) ] = tr2( Lt , S^T(q) · R )
    !      Lt = D_kk Γ^Y                 只依赖外点 k            （代码里 Lss、Lsa、Las、Laa）
    !      R  = Γbar^X S(l_q) Δ^Y B_n    只依赖内点 l            （代码里 Rss、Rsa、Ras、Raa）
    !      S^T(q)、F(rin)、F(rout)        同时依赖 k 和 l，只能在双重循环里算
    !   tr2(a, b) = Tr[a b]（见 _libraries/dirac.f90）。
    !
    ! 【矩阵怎么排】先算 Kfull(出射道, 外点径向 i, 外点角度 a, 入射道, 内点径向 jr, 内点 z jz)：
    !   内点的 y 已经积掉（振幅只依赖 |l| 和 z，y 只出现在核里）。最后
    !      Kmat = Pout · Kfull · Ein
    !   把外点角度换成出射矩、把内点 z 换成入射矩，得到 (nch·nr·nm) × (nch·nr·nm) 的方阵。
    !
    ! 【并行】外点 (i, a) 按 MPI_distribute 轮流分给各进程，各进程只填自己那部分 Kfull，
    !   最后 MPI_reducesum 相加。串行编译时 MPI_distribute 恒为真。
    ! ====================================================================================================
    !kernel matrix Kmat (ndim x ndim), psi = Kmat psi at the bound state
    subroutine faddeev_equation_kernel(this,M,Kmat)
        class(faddeev_equation):: this
        real(8):: M
        complex(8),allocatable:: Kmat(:,:)
        complex(8),allocatable,target:: Kfull(:,:,:,:,:,:)
        complex(8),pointer:: kp(:)
        complex(8),allocatable:: Rss(:,:,:,:),Rsa(:,:,:,:,:),Ras(:,:,:,:,:),Raa(:,:,:,:,:,:),Sl(:,:,:)
        complex(8):: P(4),l(4),lq(4),lqq(4),k(4),kq(4),kqq(4),q(4),rin(4),rout(4),SexT(4,4),U(4,4),D1(4,4)
        complex(8):: inS(4,4,2),inA(4,4,4,6),tmp(4,4),Lss(4,4,2),Lsa(4,4,2,4),Las(4,4,6,4),Laa(4,4,6,4,4)
        complex(8):: g(2),pref,Fin(2),Fout(2),val
        integer:: i,a,ip,n,kk,mu,rho,nu,jr,jz,ndim,ci,cj,mm,nn,b,job
        integer:: nr,nz,nza,nm,nch
        ! 几个常用维数的局部副本；g(1) = g_{0+}，g(2) = g_{1+}。P 为静止系的核子动量，M 是当前试探质量。
        nr = this%nr; nz = this%nz; nza = this%nza; nm = this%nm; nch = this%nch
        g = [this%gs,this%ga]
        P = [Zero,Zero,Zero,Im*M]
        allocate(Kfull(nch,nr,nza,nch,nr,nz)); Kfull = Zero
        ! ---------- 第 1 步：只依赖内点 l 的部分 R = Γbar^X · S(l_q) · Δ^Y · B^Y_n（每个 M 算一次）----------
        !   Sl   = S(l_q)，旁观夸克传播子（代数形式，复动量下可直接算）
        !   inS  = Δ^{0+}(l_qq^2) · Bs_n            入射标量双夸克：传播子是标量
        !   inA  = Σ_ν Δ^{1+}_{ρν}(l_qq) · Ba_{ν,n}  入射轴矢双夸克：Δ_{ρν} = (δ_{ρν} + K_ρ K_ν/m^2) F/m^2
        !   Rss(n)        = bar_s · Sl · inS(n)              X = s, Y = s
        !   Rsa(n,ρ)      = bar_s · Sl · inA(ρ,n)            X = s, Y = a
        !   Ras(n,μ)      = bar_a(μ) · Sl · inS(n)           X = a, Y = s
        !   Raa(n,μ,ρ)    = bar_a(μ) · Sl · inA(ρ,n)         X = a, Y = a
        !   注意：这里的循环变量 nu 是 Lorentz 指标（与后面第 218 行把 nu 当 y 指标用无关）。
        !inner-only pieces: Rin[XY](n,mu,rho) = Gammabar^X_mu S(l_q) Delta^Y psi_Y basis(n, rho)
        allocate(Sl(4,4,this%ni),Rss(4,4,2,this%ni),Rsa(4,4,6,4,this%ni),Ras(4,4,2,4,this%ni),Raa(4,4,6,4,4,this%ni))
        do ip=1,this%ni
            l = this%lvec(:,ip); lq = l+P/3; lqq = -l+2*P/3
            Sl(:,:,ip) = propagator(lq)
            do n=1,2
                inS(:,:,n) = Dprop0(dot4(lqq,lqq))*this%Bs_l(:,:,n,ip)
            end do
            D1 = Dprop1mat(lqq)
            do n=1,6
                do rho=1,4
                    inA(:,:,rho,n) = Zero
                    do nu=1,4
                        inA(:,:,rho,n) = inA(:,:,rho,n)+D1(rho,nu)*this%Ba_l(:,:,nu,n,ip)
                    end do
                end do
            end do
            tmp = matmul(bar_s,Sl(:,:,ip))
            do n=1,2
                Rss(:,:,n,ip) = matmul(tmp,inS(:,:,n))
            end do
            do n=1,6; do rho=1,4
                Rsa(:,:,n,rho,ip) = matmul(tmp,inA(:,:,rho,n))
            end do; end do
            do mu=1,4
                tmp = matmul(bar_a(:,:,mu),Sl(:,:,ip))
                do n=1,2
                    Ras(:,:,n,mu,ip) = matmul(tmp,inS(:,:,n))
                end do
                do n=1,6; do rho=1,4
                    Raa(:,:,n,mu,rho,ip) = matmul(tmp,inA(:,:,rho,n))
                end do; end do
            end do
        end do
        ! ---------- 第 2 步：对外点 (i, a) 循环（MPI 在这一层分配）----------
        job = 0
        do i=1,nr
            do a=1,nza
                job = job+1
                if(.not.MPI_distribute(job)) cycle
                ! 外点动量：k_q = k + P/3（出射旁观夸克），k_qq = -k + 2P/3（出射双夸克）。
                k = this%kvec(:,i,a); kq = k+P/3; kqq = -k+2*P/3
                ! 只依赖外点的部分 Lt = D^X_kk · Γ^Y（对偶基已经“转置”好，取系数直接用 tr2）：
                !   Lss(kk)      = Dts(kk) · gam_s          Lsa(kk,ρ)    = Dts(kk) · gam_a(ρ)
                !   Las(kk,μ)    = Dta(μ,kk) · gam_s        Laa(kk,μ,ρ)  = Dta(μ,kk) · gam_a(ρ)
                !outer-only pieces: Lt = Dt^X Gamma^Y_rho
                do kk=1,2
                    Lss(:,:,kk) = matmul(this%Dts(:,:,kk,i,a),gam_s)
                    do rho=1,4
                        Lsa(:,:,kk,rho) = matmul(this%Dts(:,:,kk,i,a),gam_a(:,:,rho))
                    end do
                end do
                do kk=1,6; do mu=1,4
                    Las(:,:,kk,mu) = matmul(this%Dta(:,:,mu,kk,i,a),gam_s)
                    do rho=1,4
                        Laa(:,:,kk,mu,rho) = matmul(this%Dta(:,:,mu,kk,i,a),gam_a(:,:,rho))
                    end do
                end do; end do
                ! ---------- 第 3 步：对内点 l 循环（顺序与 init 里 lvec 的存放顺序一致：径向 jr、z 节点 jz、y）----------
                !   注意：这里的循环变量 nu 只是 y 节点的编号（沿用了变量名），不是 Lorentz 指标。
                ip = 0
                do jr=1,nr
                    do jz=1,nz
                        do nu=1,this%ny                               !nu = y index here
                            ip = ip+1
                            ! 内点动量：l_q = l + P/3（入射旁观夸克），l_qq = -l + 2P/3（入射双夸克）。
                            l = this%lvec(:,ip); lq = l+P/3; lqq = -l+2*P/3
                            ! 被交换的夸克：动量 q = l_qq - k_q（入射双夸克拆出出射夸克 k_q 后剩下的那个），
                            ! 以转置传播子 S^T(q) 出现——它的 Dirac 指标流向与另外两条夸克线相反。
                            q = lqq-kq
                            SexT = transpose(propagator(q))
                            ! 双夸克振幅的标量因子 F(相对动量^2 / ω^2)，ω^2 = m^2/2：
                            !   Fin  入射双夸克 Y 在 Γ^Y 处拆开，相对动量 rin = k_q - l_qq/2
                            !   Fout 出射双夸克 X 在 Γbar^X 处形成，相对动量 rout = l_q - k_qq/2
                            !   每个数组的第 1 个元素用标量双夸克的宽度，第 2 个用轴矢双夸克的宽度。
                            rin = kq-lqq/2; rout = lq-kqq/2
                            Fin  = [Famp('s',dot4(rin,rin)),Famp('a',dot4(rin,rin))]
                            Fout = [Famp('s',dot4(rout,rout)),Famp('a',dot4(rout,rout))]
                            ! ---- X = s, Y = s：标量 -> 标量 ----
                            !   pref = -4 c_ss g_s g_s · F_s(rin) · F_s(rout) · 权重
                            !   对入射基矢 n = 1,2 和出射基矢 kk = 1,2：Kfull += pref · tr[ Lss(kk) · S^T(q) · Rss(n) ]
                            !   其余三个块（sa、as、aa）是同一个模式，只是多了 Lorentz 指标 μ（出射）、ρ（入射）的求和。
                            if(this%use_s)then                        !X = s, Y = s
                                pref = -4*this%c(1,1)*g(1)*g(1)*Fin(1)*Fout(1)*this%wl(ip)
                                do n=1,2
                                    U = matmul(SexT,Rss(:,:,n,ip))
                                    do kk=1,2
                                        val = tr2(Lss(:,:,kk),U)
                                        Kfull(this%off(1)+kk,i,a,this%off(1)+n,jr,jz) = &
                                            Kfull(this%off(1)+kk,i,a,this%off(1)+n,jr,jz)+pref*val
                                    end do
                                end do
                            end if
                            ! ---- X = s, Y = a（轴矢 -> 标量）和 X = a, Y = s（标量 -> 轴矢）----
                            !   注意 F 的取法：入射道用 Fin(Y)，出射道用 Fout(X)。
                            !   sa：对入射轴矢基矢 n = 1..6 和指标 ρ 求和；as：对出射指标 μ 求和（μ 在 Ras 和 Las 里成对出现）。
                            if(this%use_s.and.this%use_a)then         !X = s, Y = a
                                pref = -4*this%c(1,2)*g(1)*g(2)*Fin(2)*Fout(1)*this%wl(ip)
                                do n=1,6
                                    do rho=1,4
                                        U = matmul(SexT,Rsa(:,:,n,rho,ip))
                                        do kk=1,2
                                            val = tr2(Lsa(:,:,kk,rho),U)
                                            Kfull(this%off(1)+kk,i,a,this%off(2)+n,jr,jz) = &
                                                Kfull(this%off(1)+kk,i,a,this%off(2)+n,jr,jz)+pref*val
                                        end do
                                    end do
                                end do
                                pref = -4*this%c(2,1)*g(2)*g(1)*Fin(1)*Fout(2)*this%wl(ip)      !X = a, Y = s
                                do n=1,2
                                    do mu=1,4
                                        U = matmul(SexT,Ras(:,:,n,mu,ip))
                                        do kk=1,6
                                            val = tr2(Las(:,:,kk,mu),U)
                                            Kfull(this%off(2)+kk,i,a,this%off(1)+n,jr,jz) = &
                                                Kfull(this%off(2)+kk,i,a,this%off(1)+n,jr,jz)+pref*val
                                        end do
                                    end do
                                end do
                            end if
                            ! ---- X = a, Y = a：轴矢 -> 轴矢，对 μ、ρ 都求和 ----
                            if(this%use_a)then                        !X = a, Y = a
                                pref = -4*this%c(2,2)*g(2)*g(2)*Fin(2)*Fout(2)*this%wl(ip)
                                do n=1,6
                                    do mu=1,4
                                        do rho=1,4
                                            U = matmul(SexT,Raa(:,:,n,mu,rho,ip))
                                            do kk=1,6
                                                val = tr2(Laa(:,:,kk,mu,rho),U)
                                                Kfull(this%off(2)+kk,i,a,this%off(2)+n,jr,jz) = &
                                                    Kfull(this%off(2)+kk,i,a,this%off(2)+n,jr,jz)+pref*val
                                            end do
                                        end do
                                    end do
                                end do
                            end if
                        end do
                    end do
                end do
            end do
        end do
        ! ---------- 第 4 步：MPI 各进程的 Kfull 相加（kp 是指向 Kfull 的一维指针）----------
        kp(1:size(Kfull)) => Kfull
        call MPI_reducesum(kp)
        ! ---------- 第 5 步：外点角度值 -> 出射矩（Pout），入射矩 -> 内点 z 值（Ein）----------
        !   Kmat(出射 (c_i, i, m), 入射 (c_j, jr, n)) = Σ_a Σ_b Pout(m,a) · Kfull(c_i,i,a, c_j,jr,b) · Ein(b,n)
        !   行、列的一维编号与未知数向量一致：idx = m + nm*(i-1) + nm*nr*(c-1)。
        !   Nystrom 模式下 Pout、Ein 是单位矩阵，这一步只是重新编号。
        !external z-nodes -> moments (Pout), inner z-nodes <- moments (Ein)
        ndim = nch*nr*nm
        if(allocated(Kmat)) deallocate(Kmat)
        allocate(Kmat(ndim,ndim)); Kmat = Zero
        do ci=1,nch; do i=1,nr; do mm=1,nm
            do cj=1,nch; do jr=1,nr; do nn=1,nm
                val = Zero
                do a=1,nza
                    do b=1,nz
                        val = val+this%Pout(mm,a)*Kfull(ci,i,a,cj,jr,b)*this%Ein(b,nn)
                    end do
                end do
                Kmat(mm+nm*(i-1)+nm*nr*(ci-1),nn+nm*(jr-1)+nm*nr*(cj-1)) = val
            end do; end do; end do
        end do; end do; end do
    end subroutine

    ! ====================================================================================================
    ! eigen：组装 K(M)，用 LAPACK zgeev 求全部本征值，取实部最大的那个 λ(M) 及其右本征向量。
    !   束缚态（基态）在 λ(M) = 1 处；本征向量就是振幅的矩（整体的大小和相位任意）。
    ! ====================================================================================================
    subroutine faddeev_equation_eigen(this,M,lam,vec)
        class(faddeev_equation):: this
        real(8):: M
        complex(8):: lam
        complex(8),allocatable:: vec(:),Kmat(:,:)
        call this%kernel(M,Kmat)
        if(allocated(vec)) deallocate(vec)
        allocate(vec(size(Kmat,1)))
        call eig_maxreal(size(Kmat,1),Kmat,lam,vec)
    end subroutine

    ! ====================================================================================================
    ! solve：割线法解 λ(M) - 1 = 0（只看实部），初值 M0、M1 取自参数文件，容差 para%tol。
    !   收敛后保存质量 M，并用最后一次的本征向量调用 store()。
    !   每一步打印 M 和 λ - 1（只在 0 号进程打印）。
    ! ====================================================================================================
    !secant search for lambda(M) = 1; stores mass and amplitude moments
    subroutine faddeev_equation_solve(this)
        class(faddeev_equation):: this
        real(8):: M0,M1,M2,f0,f1
        complex(8):: lam
        complex(8),allocatable:: vec(:)
        integer:: it
        M0 = this%para%M0; M1 = this%para%M1
        call this%eigen(M0,lam,vec); f0 = real(lam)-1
        call this%eigen(M1,lam,vec); f1 = real(lam)-1
        do it=1,30
            M2 = M1-f1*(M1-M0)/(f1-f0)
            M0 = M1; f0 = f1
            call this%eigen(M2,lam,vec)
            M1 = M2; f1 = real(lam)-1
            if(jrank==0) print '(a,f12.7,a,es10.2)','   M =',M1,' GeV   lambda - 1 =',f1
            if(abs(f1)<this%para%tol) exit
        end do
        this%M = M1
        call this%store(vec)
    end subroutine

    ! ====================================================================================================
    ! store：把本征向量整理成振幅矩 f(c, n, i)，供之后在任意动量处取值。
    !   1. 按 idx 规则拆成 full(道 c, 径向 i, 矩 m)；只开轴矢道时 c0 = 2，使轴矢仍放在第 3..8 号道，
    !      没开的道补零（这样后面总是 8 个道）。
    !   2. Nystrom 模式下未知数是 z 节点上的值，先用 Gauss-Chebyshev 换成矩（Pm 与 Pout 同形）。
    !   3. 本征向量的整体大小和相位是任意的：除以 s1 道（只开轴矢时用 a1 道）第 0 个矩在第 1 个径向点的值。
    !      真正的归一化（重子数 = 1）在算流时另做（zero_normalization）。
    !   4. 对每个 (c, n) 在 log|l| 上做三次样条（spline_setup），供 sfun 插值。
    ! ====================================================================================================
    !amplitude moments f(c,n,i), normalised by s1 moment 0 at the first radial node = 1
    subroutine faddeev_equation_store(this,vec)
        class(faddeev_equation):: this
        complex(8):: vec(:)
        complex(8),allocatable:: full(:,:,:),mom(:,:,:)
        real(8),allocatable:: Pm(:,:)
        integer:: c,i,m,nn,a,c0
        complex(8):: ref
        allocate(full(8,this%nr,this%nm)); full = Zero
        c0 = merge(0,2,this%use_s)
        do c=1,this%nch
            do i=1,this%nr
                do m=1,this%nm
                    full(c0+c,i,m) = vec(m+this%nm*(i-1)+this%nm*this%nr*(c-1))
                end do
            end do
        end do
        if(this%para%ncheb<=0)then                     !node values -> moments
            allocate(Pm(this%nz,this%nz),mom(8,this%nr,this%nz))
            do a=1,this%nz
                Pm(:,a) = 2/pi*this%wz(a)*real(cheb_u(this%nz,dcmplx(this%z(a),0d0)))
            end do
            do c=1,8; do i=1,this%nr; do nn=1,this%nz
                mom(c,i,nn) = sum(Pm(nn,:)*full(c,i,:))
            end do; end do; end do
            full = mom
        end if
        ref = merge(full(1,1,1),full(3,1,1),this%use_s)
        if(allocated(this%f)) deallocate(this%f,this%fy,this%f2,this%logr)
        allocate(this%f(8,this%nm,this%nr),this%fy(this%nr,8*this%nm),this%f2(this%nr,8*this%nm),this%logr(this%nr))
        do c=1,8; do m=1,this%nm
            this%f(c,m,:) = full(c,:,m)/ref
            this%fy(:,c+8*(m-1)) = this%f(c,m,:)
        end do; end do
        this%logr = log(this%r)
        call spline_setup(this%nr,this%logr,8*this%nm,this%fy,this%f2)
    end subroutine

    ! ====================================================================================================
    ! sfun：任意（可以是复的）l 处的 8 个标量函数 f_c(l^2, l·P)。
    !   径向：在 log|l| 上做样条插值。|l| 超出 [r_1, r_nr] 时被截到端点，即网格外振幅“冻结”为端点值——
    !         对紫外敏感的量（高阶 PDF 矩等）要加截断，见学习笔记“常见的坑”。
    !   角度：zz = -i l·Phat/|l|，静止系下就是 l_4/|l|；P boost 以后 zz 变成复数，Chebyshev 多项式 U_n 照样能算，
    !         这就是振幅的解析延拓。
    ! ====================================================================================================
    !the 8 scalar functions f_c(l^2, l.P) for arbitrary l and on-shell P (P^2 = -M^2)
    function faddeev_equation_sfun(this,l,P) result(fc)
        class(faddeev_equation):: this
        complex(8):: l(4),P(4),fc(8),Phat(4),L2,zz,u(this%nm),v(8*this%nm)
        real(8):: t
        integer:: c
        Phat = P/this%M
        L2 = sqrt(dot4(l,l))
        zz = -Im*dot4(l,Phat)/L2
        t = min(max(log(real(L2)),this%logr(1)),this%logr(this%nr))
        v = spline_eval(this%nr,this%logr,8*this%nm,this%fy,this%f2,t)
        u = cheb_u(this%nm,zz)
        do c=1,8
            fc(c) = sum(v(c:8*this%nm:8)*u)
        end do
    end function

    ! ====================================================================================================
    ! amplitude：任意 l、P 处的完整振幅矩阵（Λ+ 已右乘）：
    !   Ps     = [ f_1 S^1 + f_2 S^2 ] Λ+(P)                      标量双夸克道，4×4（夸克指标 × 核子指标）
    !   Pa_ν   = [ Σ_{n=1..6} f_{2+n} γ5 A^n_ν ] Λ+(P)             轴矢双夸克道，多一个 Lorentz 指标 ν
    ! ====================================================================================================
    !Psi_s(l;P) Lambda_+(P) and Psi_a,nu(l;P) Lambda_+(P)
    subroutine faddeev_equation_amplitude(this,l,P,Ps,Pa)
        class(faddeev_equation):: this
        complex(8):: l(4),P(4),Ps(4,4),Pa(4,4,4),fc(8),lhat(4),Bs(4,4,2),Ba(4,4,4,6),Lp(4,4)
        integer:: n,nu
        fc = this%sfun(l,P)
        lhat = l/real(sqrt(dot4(l,l)))
        call basis(lhat,P/this%M,Bs,Ba)
        Lp = lam_plus(P,this%M)
        Ps = matmul(fc(1)*Bs(:,:,1)+fc(2)*Bs(:,:,2),Lp)
        do nu=1,4
            Pa(:,:,nu) = Zero
            do n=1,6
                Pa(:,:,nu) = Pa(:,:,nu)+fc(2+n)*Ba(:,:,nu,n)
            end do
            Pa(:,:,nu) = matmul(Pa(:,:,nu),Lp)
        end do
    end subroutine

    ! ====================================================================================================
    ! conj_amplitude：共轭振幅 Ψbar(p; -P) = C† Ψ(-p; -P)^T C，算流（Ψbar ... Ψ）时用。
    !   实现上：先在 (-p, -Pf) 处取振幅，再做 C† (·)^T C。
    ! ====================================================================================================
    !Psibar(p;-P) = C^dag Psi(-p;-P)^T C
    subroutine faddeev_equation_conj_amplitude(this,p,Pf,Cs,Ca)
        class(faddeev_equation):: this
        complex(8):: p(4),Pf(4),Cs(4,4),Ca(4,4,4),Ps(4,4),Pa(4,4,4)
        integer:: nu
        call this%amplitude(-p,-Pf,Ps,Pa)
        Cs = matmul(ccdag,matmul(transpose(Ps),cc))
        do nu=1,4
            Ca(:,:,nu) = matmul(ccdag,matmul(transpose(Pa(:,:,nu)),cc))
        end do
    end subroutine

    ! ====================================================================================================
    ! output：把振幅矩写到 outdata/<root>/amplitude.txt（只在 0 号进程写）。
    !   每行：|l|、道 c（1,2 = s1,s2；3..8 = a1..a6）、矩序号 n、Re f、Im f。
    ! ====================================================================================================
    subroutine faddeev_equation_output(this)
        class(faddeev_equation):: this
        integer:: c,m,i
        if(jrank/=0) return
        call x_mkdir('outdata/'//trim(this%para%root))
        open(unit=101,file='outdata/'//trim(this%para%root)//'/amplitude.txt',status='replace')
        write(101,'(a,f12.7)') '# M_N [GeV] = ',this%M
        write(101,'(a)') '# columns: |l| [GeV], channel c (1,2 = s1,s2; 3..8 = a1..a6), moment n, Re f, Im f'
        do i=1,this%nr; do c=1,8; do m=1,this%nm
            write(101,'(es16.8,2i4,2es18.9)') this%r(i),c,m-1,this%f(c,m,i)
        end do; end do; end do
        close(101)
    end subroutine

end module
