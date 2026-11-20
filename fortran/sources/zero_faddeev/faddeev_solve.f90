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
module faddeev_solve
    use libraries
    use quark_elems
    use diquark_elems
    use faddeev_elems
    use faddeev_utils
    implicit none

private

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

    subroutine faddeev_equation_init(this,para)
        class(faddeev_equation):: this
        type(faddeev_para):: para
        integer:: i,a,j,b,iy,p,n
        real(8):: sz,sy
        complex(8):: lhat(4),Phat(4),Bs(4,4,2),Ba(4,4,4,6),Lp(4,4)
        call this%free()
        this%para = para
        this%gs = para%gs; this%ga = para%ga
        this%use_s = index(para%channels,'s')>0
        this%use_a = index(para%channels,'a')>0
        this%nch = 0
        if(this%use_s)then
            this%off(1) = 0; this%nch = 2
        end if
        if(this%use_a)then
            this%off(2) = this%nch; this%nch = this%nch+6
        end if
        this%c = flavour_coeffs([1d0,0d0])
        this%nr = para%nr; this%nz = para%nz; this%ny = para%ny
        allocate(this%r(this%nr),this%wr(this%nr),this%z(this%nz),this%wz(this%nz),this%y(this%ny),this%wy(this%ny))
        call radial(this%nr,para%scale,this%r,this%wr)
        call chebyshev_u(this%nz,this%z,this%wz)
        call gauleg(this%ny,this%y,this%wy)
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

    subroutine faddeev_equation_free(this)
        class(faddeev_equation):: this
        if(allocated(this%r)) deallocate(this%r,this%wr,this%z,this%wz,this%y,this%wy)
        if(allocated(this%zext)) deallocate(this%zext,this%wext,this%Pout,this%Ein)
        if(allocated(this%kvec)) deallocate(this%kvec,this%Dts,this%Dta)
        if(allocated(this%lvec)) deallocate(this%lvec,this%wl,this%Bs_l,this%Ba_l)
        if(allocated(this%f)) deallocate(this%f,this%fy,this%f2,this%logr)
    end subroutine

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
        nr = this%nr; nz = this%nz; nza = this%nza; nm = this%nm; nch = this%nch
        g = [this%gs,this%ga]
        P = [Zero,Zero,Zero,Im*M]
        allocate(Kfull(nch,nr,nza,nch,nr,nz)); Kfull = Zero
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
        job = 0
        do i=1,nr
            do a=1,nza
                job = job+1
                if(.not.MPI_distribute(job)) cycle
                k = this%kvec(:,i,a); kq = k+P/3; kqq = -k+2*P/3
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
                ip = 0
                do jr=1,nr
                    do jz=1,nz
                        do nu=1,this%ny                               !nu = y index here
                            ip = ip+1
                            l = this%lvec(:,ip); lq = l+P/3; lqq = -l+2*P/3
                            q = lqq-kq
                            SexT = transpose(propagator(q))
                            rin = kq-lqq/2; rout = lq-kqq/2
                            Fin  = [Famp('s',dot4(rin,rin)),Famp('a',dot4(rin,rin))]
                            Fout = [Famp('s',dot4(rout,rout)),Famp('a',dot4(rout,rout))]
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
        kp(1:size(Kfull)) => Kfull
        call MPI_reducesum(kp)
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
