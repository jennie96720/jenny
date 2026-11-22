!Light-front Mellin moments of the nucleon DFs: shared elements (= qdq/pdf.py, qdq/pdf_pol.py)
!
!  <x^m>_q = Tr[Y n.J^(m)_q] / normalisation,  n = (0,0,1,i) (n.n = 0), rest frame P = (0,0,0,iM)
!  each forward (Q = 0) insertion weighted by w_m = [x/(1 + lam p^2)]^m, x = n.p/n.P, p the struck line
!  (lam = 0: plain moment; lam > 0: regulator of Bednar, Cloet, Tandy, arXiv:1803.03656, extrapolated
!  to lam -> 0 in the Python post-processing, scripts/15_fortran_post.py).
!  Y = 1 (number current) or Y = T^dagger, T = Lambda_+ gamma_5 gamma.n Lambda_+ (helicity).
module pdf_elems
    use libraries
    use quark_elems
    use diquark_elems
    use faddeev_elems
    use current1_elems
    implicit none

type pdf_para
    integer:: mmax = 8                                  !moments m = 0..mmax
    integer:: nlam = 1                                  !number of regulator values
    real(8):: lams(32) = 0d0                            !regulator values lam [GeV^-2], first nlam used
    real(8):: kmax = 15d0                               !UV cutoff on |k| (one- and two-loop grids) [GeV]
    integer:: nk = 48, nz = 32, ny = 24                 !one-loop grid
    integer:: nko(3) = [12,6,6]                         !two-loop outer grid (|k|, z, y)
    integer:: nli(4) = [12,6,6,6]                       !two-loop inner grid (|l|, z, y, phi)
    real(8):: scale = 0.6d0
    real(8):: eta = 1d0/3d0
    logical:: two_loop = .true.                         !.false.: diagrams 1-3 only (fast checks)
    !diquark axial couplings (helicity): table in the complex K^2 plane
    integer:: tq(4) = [64,24,16,8]                      !quark-loop grid (|q|, z, y, phi)
    integer:: nre = 12, nim = 7
    real(8):: re(32) = 0d0, im(32) = 0d0                !Re K^2, Im K^2 >= 0 nodes [GeV^2]
    character(len=80):: table = 'outdata/pdf/axial_couplings.txt'
    !diquark leading-twist DA moments
    integer:: da(3) = [128,64,14]                       !grid (|k|, z, y), scale 1 GeV
    real(8):: da_kmax = 20d0
    character(len=40):: root = 'pdf'
end type pdf_para

    complex(8),parameter:: nvec(4) = [Zero,Zero,One,Im]

public

contains

    !w(l,m) = [x/(1 + lam_l p2)]^m, m = 0..mmax
    function lf_weights(x,p2,lams,nlam,mmax) result(w)
        complex(8),intent(in):: x,p2
        integer,intent(in):: nlam,mmax
        real(8),intent(in):: lams(nlam)
        complex(8):: w(nlam,0:mmax),r
        integer:: l,m
        do l=1,nlam
            r = x/(1+lams(l)*p2)
            w(l,0) = One
            do m=1,mmax
                w(l,m) = w(l,m-1)*r
            end do
        end do
    end function

    !A(p^2) of S^{-1} = i gamma.p A + B:  A = sigma_V/(p^2 sigma_V^2 + sigma_S^2)
    complex(8) function A_of(p2)
        complex(8):: p2,sv,ss
        call sigmas(p2,sv,ss)
        A_of = sv/(p2*sv**2+ss**2)
    end function

    !unamputated forward axial vertex S(l) gamma_5[gamma_mu A + 2 gamma.l l_mu A'] S(l)   (Chen et al.
    !2103.02054 Eq. (26) at Q = 0; the pion-pole term vanishes there) -> (4,4,mu)
    function axial_quark(l) result(v)
        complex(8),intent(in):: l(4)
        complex(8):: v(4,4,4),p2,A,dA,S(4,4),g5l(4,4)
        real(8),parameter:: h = 1d-6
        integer:: mu
        p2 = dot4(l,l)
        A = A_of(p2)
        dA = (A_of(p2+h)-A_of(p2-h))/(2*h)
        S = propagator(l)
        g5l = matmul(g5,slash(l))
        do mu=1,4
            v(:,:,mu) = matmul(S,matmul(A*matmul(g5,gm(:,:,mu))+2*l(mu)*dA*g5l,S))
        end do
    end function

    !T = Lambda_+ gamma_5 gamma.n Lambda_+,  TV = Lambda_+ gamma.n Lambda_+,  Xp = T^dagger
    subroutine lf_structures(P,M,T,TV,Xp)
        complex(8),intent(in):: P(4)
        real(8),intent(in):: M
        complex(8),intent(out):: T(4,4),TV(4,4),Xp(4,4)
        complex(8):: L(4,4)
        L = lam_plus(P,M)
        T = matmul(L,matmul(g5,matmul(slash(nvec),L)))
        TV = matmul(L,matmul(slash(nvec),L))
        Xp = conjg(transpose(T))
    end subroutine

    function pdf_para_load(fstr) result(para)
        character(len=*):: fstr
        type(pdf_para):: para
        namelist /pdf_para_list/ para
        open(unit=100,file=trim(fstr),status='old')
        read(unit=100,nml=pdf_para_list)
        close(unit=100)
    end function

    !keep the grid points with |k| < kmax (Euclidean norm of the real loop momentum)
    subroutine compact_grid(gr,kmax)
        type(current_grid),intent(inout):: gr
        real(8),intent(in):: kmax
        logical:: keep(gr%n)
        complex(8),allocatable:: k(:,:)
        real(8),allocatable:: w(:)
        integer:: i
        do i=1,gr%n
            keep(i) = sqrt(sum(real(gr%k(:,i))**2))<kmax
        end do
        k = gr%k(:,pack([(i,i=1,gr%n)],keep))
        w = pack(gr%w,keep)
        gr%n = size(w)
        call move_alloc(k,gr%k); call move_alloc(w,gr%w)
    end subroutine

end module
