!External probes coupling to the quark+diquark nucleon (= qdq/current.py Vertices, qdq/probes.py)
!
!probe index m:  photon m = mu (1..4);  graviton m = 4(mu-1)+nu (1..16)
!  quark(l1,l2)  unamputated quark vertex incl. the factor i            -> (4,4,m)
!  dq_s(l1,l2)   unamputated scalar-diquark vertex                       -> (m)
!  dq_a(l1,l2)   unamputated axial-diquark vertex                        -> (alpha,beta,m)
!  trans(l1,l2)  amputated 0+ <-> 1+ transition vertex                   -> (alpha,m)
!WGTI: photon  Q.V = G(in) - G(out);  graviton  Q_mu V_{mu nu} = l_in,nu G(in) - G(out) l_out,nu
!
!Photon: Cheng et al. arXiv:2507.13484 Eqs. (24), (27), (29), (31), Table 1.
!Graviton: arXiv:2611.19819 Eqs. (14)-(27): EM vertex x l_nu - delta_{mu nu} x mean propagator
!          (+ scalar-meson pole for the quark); diquark/transition flavour factor = average over its
!          two quarks (half the EM-style sum), as the EMT Ward identity requires.
!Transition vertex: chi^{10} = +chi^{01}, kappa = 2.64 x (-i) in OUR gamma conventions (derived from the
!          quark loop inside the diquark, see qdq/current.py transition_kappas).
module probe_elems
    use libraries
    use quark_elems
    use diquark_elems
    use faddeev_elems
    implicit none

type probe_type
    character(len=4):: kind = 'em'
    integer:: nidx = 4
    real(8):: M = 1.18d0                                        !nucleon mass (transition vertex, projectors)
    real(8):: a(3) = [0.17d0,1.47d0,-0.93d0], b(3) = [1.03d0,-0.17d0,1.61d0]
    real(8):: MEq = 0.33d0                                      !M_q^E in the transverse vertex
    real(8):: r_sc = 0.31d0, r_av = 1.05d0                      !diquark radii [fm]
    real(8):: kappa01 = 2.64d0
    complex(8):: kphase = (0d0,-1d0)
    real(8):: m_S = 0.55d0, r_S = 0.193d0                       !scalar-meson pole (graviton only)
contains
    procedure,public:: quark  => probe_quark
    procedure,public:: dq_s   => probe_dq_s
    procedure,public:: dq_a   => probe_dq_a
    procedure,public:: trans  => probe_trans
    procedure,public:: lift   => probe_lift
    procedure,public:: projectors => probe_projectors
end type probe_type

public

contains

    function probe_create(kind,M) result(pr)
        character(len=*):: kind
        real(8):: M
        type(probe_type):: pr
        pr%kind = kind; pr%M = M
        if(kind=='grav')then
            pr%nidx = 16; pr%r_sc = 0.23d0; pr%r_av = 0.78d0
        else
            pr%nidx = 4
        end if
    end function

    !--------------------------- elementary pieces ---------------------------
    ![f(x1)-f(x2)]/(x1-x2) with a central derivative when x1 ~ x2;  f selected by id
    complex(8) function divdiff(id,x1,x2)
        integer:: id
        complex(8):: x1,x2,d,xm
        real(8),parameter:: h = 1d-6
        d = x1-x2
        if(abs(d)<1d-9*(1+abs(x1)))then
            xm = 0.5d0*(x1+x2)
            divdiff = (fsel(id,xm+h)-fsel(id,xm-h))/(2*h)
        else
            divdiff = (fsel(id,x1)-fsel(id,x2))/d
        end if
    end function

    complex(8) function fsel(id,x)
        integer:: id
        complex(8):: x
        select case(id)
        case(1); fsel = sigV(x)
        case(2); fsel = sigS(x)
        case(3); fsel = Dprop0(x)
        case(4); fsel = Dprop1(x)
        case(5); fsel = Famp('s',x)
        case(6); fsel = Famp('a',x)
        case default; fsel = Zero
        end select
    end function

    complex(8) function Fj(Q2,r_fm)
        complex(8):: Q2
        real(8):: r_fm
        Fj = 1/(1+(r_fm/hbarc)**2*Q2/6)
    end function

    !EM quark vertex chi_mu WITHOUT the factor i:  i (l1-l2).chi = S(l2) - S(l1)
    function chi_em(this,l1,l2) result(chi)
        class(probe_type):: this
        complex(8):: l1(4),l2(4),chi(4,4,4)
        complex(8):: a1,a2,SigVv,DV,DS,l(4),lc(4),lc2,E,s(3),gl(4,4),glc(4,4),slc(4,4,4)
        integer:: mu,nu
        a1 = dot4(l1,l1); a2 = dot4(l2,l2)
        SigVv = 0.5d0*(sigV(a1)+sigV(a2))
        DV = divdiff(1,a1,a2); DS = divdiff(2,a1,a2)
        l = 0.5d0*(l1+l2); lc = l1-l2; lc2 = dot4(lc,lc)
        E = sqrt(1+lc2/(2*this%MEq)**2)-1
        s = this%a+this%b*exp(-0.25d0*E)
        gl = slash(l); glc = slash(lc)
        do mu=1,4
            slc(:,:,mu) = Zero
            do nu=1,4
                slc(:,:,mu) = slc(:,:,mu)+sig(:,:,mu,nu)*lc(nu)
            end do
        end do
        do mu=1,4
            chi(:,:,mu) = gm(:,:,mu)*SigVv+2*l(mu)*(gl*DV+Im*DS*id4) &
                + 0.5d0*(2*s(1)-1)*DV*(matmul(glc,matmul(gm(:,:,mu),gl))-matmul(gl,matmul(gm(:,:,mu),glc))) &
                - (2*s(2)-1)*DS*slc(:,:,mu) &
                + 0.5d0*Im*(1+s(3))*DV*matmul(glc,slc(:,:,mu))
        end do
    end function

    function chi_s_em(this,l1,l2) result(v)
        class(probe_type):: this
        complex(8):: l1(4),l2(4),v(4),lc(4)
        lc = l1-l2
        v = -2*0.5d0*(l1+l2)*divdiff(3,dot4(l1,l1),dot4(l2,l2))*Fj(dot4(lc,lc),this%r_sc)
    end function

    function chi_a_em(this,l1,l2) result(chi)        !(alpha,beta,mu)
        class(probe_type):: this
        complex(8):: l1(4),l2(4),chi(4,4,4),l(4),lc(4),Fav,dD,D1a,D1b
        integer:: mu,al,be
        real(8):: m2
        l = 0.5d0*(l1+l2); lc = l1-l2
        Fav = Fj(dot4(lc,lc),this%r_av)
        dD = divdiff(4,dot4(l1,l1),dot4(l2,l2))
        D1a = Dprop1(dot4(l1,l1)); D1b = Dprop1(dot4(l2,l2))
        m2 = dq_m1**2
        do mu=1,4; do al=1,4; do be=1,4
            chi(al,be,mu) = 2*merge(One,Zero,al==be)*l(mu)*dD &
                + (merge(One,Zero,mu==al)*l1(be)*D1a+merge(One,Zero,mu==be)*l2(al)*D1b)/m2 &
                + 2/m2*l(mu)*l2(al)*l1(be)*dD
            chi(al,be,mu) = -Fav*chi(al,be,mu)
        end do; end do; end do
    end function

    function chi_01_em(this,l1,l2) result(t)          !(alpha,mu)
        class(probe_type):: this
        complex(8):: l1(4),l2(4),t(4,4),lc(4),Fav
        integer:: mu,al,r,s
        lc = l1-l2
        Fav = Fj(dot4(lc,lc),this%r_av)
        do mu=1,4; do al=1,4
            t(al,mu) = Zero
            do r=1,4; do s=1,4
                t(al,mu) = t(al,mu)+eps4(mu,al,r,s)*l1(r)*l2(s)
            end do; end do
            t(al,mu) = Im/this%M*this%kappa01*this%kphase*t(al,mu)*Fav
        end do; end do
    end function

    !--------------------------- probe interface ---------------------------
    function probe_quark(this,l1,l2) result(v)
        class(probe_type):: this
        complex(8):: l1(4),l2(4)
        complex(8),allocatable:: v(:,:,:)
        complex(8):: c(4,4,4),l(4),Q(4),Q2,S1(4,4),S2(4,4),S12(4,4),TP
        integer:: mu,nu
        allocate(v(4,4,this%nidx))
        c = Im*chi_em(this,l1,l2)
        if(this%kind=='em')then
            v = c; return
        end if
        l = 0.5d0*(l1+l2); Q = l1-l2; Q2 = dot4(Q,Q)
        S1 = propagator(l1); S2 = propagator(l2); S12 = matmul(S1,S2)
        do mu=1,4; do nu=1,4
            TP = -this%r_S*(Q2*merge(One,Zero,mu==nu)-Q(mu)*Q(nu))/(Q2+this%m_S**2)
            v(:,:,4*(mu-1)+nu) = l(nu)*c(:,:,mu)-TP*S12
            if(mu==nu) v(:,:,4*(mu-1)+nu) = v(:,:,4*(mu-1)+nu)-0.5d0*(S1+S2)
        end do; end do
    end function

    function probe_dq_s(this,l1,l2) result(v)
        class(probe_type):: this
        complex(8):: l1(4),l2(4),c(4),l(4),lc(4),sigma
        complex(8),allocatable:: v(:)
        integer:: mu,nu
        allocate(v(this%nidx))
        c = chi_s_em(this,l1,l2)
        if(this%kind=='em')then
            v = c; return
        end if
        l = 0.5d0*(l1+l2); lc = l1-l2
        sigma = 0.5d0*(Dprop0(dot4(l1,l1))+Dprop0(dot4(l2,l2)))*Fj(dot4(lc,lc),this%r_sc)
        do mu=1,4; do nu=1,4
            v(4*(mu-1)+nu) = l(nu)*c(mu)-merge(sigma,Zero,mu==nu)
        end do; end do
    end function

    function probe_dq_a(this,l1,l2) result(v)
        class(probe_type):: this
        complex(8):: l1(4),l2(4),c(4,4,4),l(4),lc(4),sigma(4,4)
        complex(8),allocatable:: v(:,:,:)
        integer:: mu,nu
        allocate(v(4,4,this%nidx))
        c = chi_a_em(this,l1,l2)
        if(this%kind=='em')then
            v = c; return
        end if
        l = 0.5d0*(l1+l2); lc = l1-l2
        sigma = 0.5d0*(Dprop1mat(l1)+Dprop1mat(l2))*Fj(dot4(lc,lc),this%r_av)
        do mu=1,4; do nu=1,4
            v(:,:,4*(mu-1)+nu) = l(nu)*c(:,:,mu)
            if(mu==nu) v(:,:,4*(mu-1)+nu) = v(:,:,4*(mu-1)+nu)-sigma
        end do; end do
    end function

    function probe_trans(this,l1,l2) result(v)
        class(probe_type):: this
        complex(8):: l1(4),l2(4),t(4,4),l(4)
        complex(8),allocatable:: v(:,:)
        integer:: mu,nu
        allocate(v(4,this%nidx))
        t = chi_01_em(this,l1,l2)
        if(this%kind=='em')then
            v = t; return
        end if
        l = 0.5d0*(l1+l2)
        do mu=1,4; do nu=1,4
            v(:,4*(mu-1)+nu) = l(nu)*t(:,mu)
        end do; end do
    end function

    !seagull: EM structure (4,4,mu) -> probe; graviton: weight_nu x vec_mu + delta_{mu nu}/2 x smid
    function probe_lift(this,vec,w,smid) result(v)
        class(probe_type):: this
        complex(8):: vec(4,4,4),w(4),smid(4,4)
        complex(8),allocatable:: v(:,:,:)
        integer:: mu,nu
        allocate(v(4,4,this%nidx))
        if(this%kind=='em')then
            v = vec; return
        end if
        do mu=1,4; do nu=1,4
            v(:,:,4*(mu-1)+nu) = vec(:,:,mu)*w(nu)
            if(mu==nu) v(:,:,4*(mu-1)+nu) = v(:,:,4*(mu-1)+nu)+0.5d0*smid
        end do; end do
    end function

    !dual structures Tt(:,:,m,b) and map from projections s_b to form factors (F = Map s)
    !EM: (F1,F2[,F3]); graviton: (A,J,D,S,cbar), or (A,cbar) at Q = 0.  Includes the phase -i (graviton).
    subroutine probe_projectors(this,Pf,Pi,Qv,Tt,Map)
        class(probe_type):: this
        complex(8):: Pf(4),Pi(4),Qv(4)
        complex(8),allocatable:: Tt(:,:,:,:),Map(:,:)
        complex(8):: Lf(4,4),Li(4,4),sQ(4,4,4),T(4,4,4,3),Gr(3,3),P(4),P2,Q2,Qh(4),gp(4,4,4),gQ(4,4)
        real(8):: M,tau
        integer:: mu,nu,nb,i,j
        M = this%M
        if(allocated(Tt)) deallocate(Tt)
        if(allocated(Map)) deallocate(Map)
        if(this%kind=='em')then
            Lf = lam_plus(Pf,M); Li = lam_plus(Pi,M)
            do mu=1,4
                sQ(:,:,mu) = Zero
                do nu=1,4
                    sQ(:,:,mu) = sQ(:,:,mu)+sig(:,:,mu,nu)*Qv(nu)
                end do
            end do
            nb = merge(1,3,all(abs(Qv)==0d0))
            allocate(Tt(4,4,4,nb),Map(nb,nb))
            do mu=1,4
                T(:,:,mu,1) = matmul(Lf,matmul(gm(:,:,mu),Li))
                T(:,:,mu,2) = matmul(Lf,matmul(sQ(:,:,mu),Li))/(2*M)
                T(:,:,mu,3) = matmul(Lf,Li)*Qv(mu)/(2*M)
                Tt(:,:,mu,1) = matmul(Li,matmul(gm(:,:,mu),Lf))
                if(nb==3)then
                    Tt(:,:,mu,2) = matmul(Li,matmul(sQ(:,:,mu),Lf))/(2*M)
                    Tt(:,:,mu,3) = matmul(Li,Lf)*Qv(mu)/(2*M)
                end if
            end do
            do i=1,nb; do j=1,nb
                Gr(i,j) = Zero
                do mu=1,4
                    Gr(i,j) = Gr(i,j)+tr2(Tt(:,:,mu,i),T(:,:,mu,j))
                end do
            end do; end do
            Map = Gr(1:nb,1:nb)
            call zinverse(nb,Map)
            return
        end if
        !graviton: 2611.19819 Eqs. (5)-(6)
        P = 0.5d0*(Pf+Pi); P2 = dot4(P,P); Q2 = dot4(Qv,Qv)
        if(abs(Q2)>0d0)then
            Qh = Qv/sqrt(Q2)
        else
            Qh = [Zero,Zero,One,Zero]           !Q -> 0 limit along the Breit-frame direction of Q
        end if
        gQ = slash(Qh)
        do mu=1,4
            gp(:,:,mu) = gm(:,:,mu)-Qh(mu)*gQ
        end do
        allocate(Tt(4,4,16,5))
        do mu=1,4; do nu=1,4
            i = 4*(mu-1)+nu
            Tt(:,:,i,1) = M/P2**2*P(mu)*P(nu)*id4
            Tt(:,:,i,2) = Im/P2*gp(:,:,mu)*P(nu)
            Tt(:,:,i,3) = M/P2*merge(One,Zero,mu==nu)*id4
            Tt(:,:,i,4) = M/P2*matmul(gp(:,:,mu),gp(:,:,nu))
            Tt(:,:,i,5) = Im/P2*P(mu)*gp(:,:,nu)
        end do; end do
        tau = real(Q2/(4*M**2))
        if(tau==0d0)then
            allocate(Map(2,5))
            Map(1,:) = [-5,-1,0,1,-1]/4d0
            Map(2,:) = [0d0,0d0,-0.5d0,0.5d0,0d0]
        else
            allocate(Map(5,5))
            Map(1,:) = [-5,-1,0,1,-1]/(4*(1+tau))
            Map(2,:) = [-2,-1,0,0,-1]/(8*tau)
            Map(3,:) = [1,0,2,-3,0]/(4*tau)
            Map(4,:) = [0,-1,0,0,1]/(8*tau)
            Map(5,:) = [0d0,0d0,-0.5d0,0.5d0,0d0]
        end if
        Map = -Im*Map
    end subroutine

    !--------------------------- flavour factors ---------------------------
    !diagrams 1-3: f1(X), f2(X) (X = 1 s, 2 a), f3(XY) (12 = s<-a, 21 = a<-s);  graviton: f2, f3 halved
    subroutine flavour_factors(this,charge,phi,f1,f2,f3)
        class(probe_type):: this
        real(8):: charge(2,2),phi(2),f1(2),f2(2),f3(2,2)
        integer:: X,Y,m,n
        f1 = 0d0; f2 = 0d0; f3 = 0d0
        do X=1,2
            do m=lo(X),hi(X)
                f1(X) = f1(X)+dot_product(phi,matmul(transpose(sf(:,:,m)),matmul(charge,matmul(sf(:,:,m),phi))))
                do n=lo(X),hi(X)
                    f2(X) = f2(X)+dot_product(phi,matmul(transpose(sf(:,:,m)),matmul(sf(:,:,n),phi))) &
                        *trace2(matmul(transpose(tf(:,:,m)),matmul(charge,tf(:,:,n))+matmul(tf(:,:,n),charge)))
                end do
            end do
        end do
        do X=1,2
            Y = 3-X
            do m=lo(X),hi(X); do n=lo(Y),hi(Y)
                f3(X,Y) = f3(X,Y)+dot_product(phi,matmul(transpose(sf(:,:,m)),matmul(sf(:,:,n),phi))) &
                    *trace2(matmul(transpose(tf(:,:,m)),matmul(charge,tf(:,:,n))-matmul(tf(:,:,n),charge)))
            end do; end do
        end do
        if(this%kind=='grav')then
            f2 = f2/2; f3 = f3/2
        end if
    end subroutine

    !diagrams 4-6: c(XY, piece), pieces 1 ex4, 2 by5, 3 ex5, 4 by6, 5 ex6
    function exchange_flavour(charge,phi) result(c)
        real(8):: charge(2,2),phi(2),c(2,2,5),ex,b5,b6,s1(2),s2(2)
        integer:: X,Y,m,n
        do X=1,2; do Y=1,2
            ex = 0d0; b5 = 0d0; b6 = 0d0
            do m=lo(X),hi(X); do n=lo(Y),hi(Y)
                s1 = matmul(sf(:,:,m),phi)               !(s^m phi)
                s2 = matmul(sf(:,:,n),phi)
                ex = ex+dot_product(s1,matmul(tf(:,:,n),matmul(transpose(charge),matmul(transpose(tf(:,:,m)),s2))))
                b5 = b5+dot_product(s1,matmul(charge,matmul(tf(:,:,n),matmul(transpose(tf(:,:,m)),s2))))
                b6 = b6+dot_product(s1,matmul(tf(:,:,n),matmul(transpose(tf(:,:,m)),matmul(charge,s2))))
            end do; end do
            c(X,Y,:) = [ex,b5,ex,b6,ex]
        end do; end do
    end function

    pure integer function lo(X)
        integer,intent(in):: X
        lo = merge(0,1,X==1)
    end function

    pure integer function hi(X)
        integer,intent(in):: X
        hi = merge(0,3,X==1)
    end function

    pure real(8) function trace2(a)
        real(8),intent(in):: a(2,2)
        trace2 = a(1,1)+a(2,2)
    end function

end module
