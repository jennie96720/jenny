!Elements of the nucleon quark+diquark Faddeev equation (= qdq/faddeev.py)
!   * flavour coefficients c_XY = sum_{m in X, n in Y} phi^T s^mT t^n t^mT s^n phi   (2611.19819 App. B)
!   * positive-energy basis S^{1,2}, gamma_5 A^{1..6}_nu (2611.19819 Eq. A15)
!   * dual basis for the least-squares projection onto the basis
module faddeev_elems
    use libraries
    use quark_elems
    use diquark_elems
    implicit none

type faddeev_para
    integer:: nr     = 16                   !radial points
    integer:: nz     = 8                    !Chebyshev nodes of z = l_4/|l|
    integer:: ny     = 8                    !Gauss-Legendre nodes of the 3-angle y
    integer:: ncheb  = 6                    !number of Chebyshev moments (0: plain Nystrom)
    real(8):: scale  = 0.6d0                !radial map |l| = s t/(1-t)
    character(len=2):: channels = 'sa'      !'s', 'a' or 'sa'
    logical:: own_g  = .false.              !.true.: couplings from outdata/diquark.txt
    complex(8):: gs  = (7.47d0,0d0)         !g_{0+}  (2611.19819 value / 2)
    complex(8):: ga  = (0d0,6.49d0)         !g_{1+}  (2611.19819 value / 2, imaginary here)
    real(8):: M0     = 1.17d0               !secant start values for lambda(M) = 1
    real(8):: M1     = 1.19d0
    real(8):: tol    = 1d-6
    character(len=40):: root = 'faddeev'
end type faddeev_para

public

contains

    pure integer function chan_size(X)
        character,intent(in):: X
        chan_size = merge(2,6,X=='s')
    end function

    !flavour coefficients c(X,Y), X,Y = 1 (0+), 2 (1+), for isospinor phi
    function flavour_coeffs(phi) result(c)
        real(8),intent(in):: phi(2)
        real(8):: c(2,2)
        integer:: X,Y,m,n,m1,m2,n1,n2
        do X=1,2
            do Y=1,2
                c(X,Y) = 0d0
                m1 = merge(0,1,X==1); m2 = merge(0,3,X==1)
                n1 = merge(0,1,Y==1); n2 = merge(0,3,Y==1)
                do m=m1,m2
                    do n=n1,n2
                        c(X,Y) = c(X,Y)+dot_product(phi,matmul(transpose(sf(:,:,m)),matmul(tf(:,:,n), &
                                 matmul(transpose(tf(:,:,m)),matmul(sf(:,:,n),phi)))))
                    end do
                end do
            end do
        end do
    end function

    function lam_plus(P,M) result(L)
        complex(8),intent(in):: P(4)
        real(8),intent(in):: M
        complex(8):: L(4,4)
        L = (-Im*slash(P)+M*id4)/(2*M)
    end function

    !Bs(:,:,n) n=1,2 and Ba(:,:,nu,n) n=1..6 (gamma_5 included) for unit lhat and Phat (Phat^2 = -1)
    subroutine basis(lhat,Phat,Bs,Ba)
        complex(8),intent(in):: lhat(4),Phat(4)
        complex(8),intent(out):: Bs(4,4,2),Ba(4,4,4,6)
        complex(8):: lP,lperp(4),gl(4,4),glp(4,4),gP(4,4),gperp(4,4,4),A(4,4,4,6)
        integer:: nu,n
        lP = dot4(lhat,Phat)
        lperp = lhat+lP*Phat
        lperp = lperp/sqrt(dot4(lperp,lperp))
        gl = slash(lhat); glp = slash(lperp); gP = slash(Phat)
        Bs(:,:,1) = id4
        Bs(:,:,2) = Im*gl-lP*id4
        do nu=1,4
            gperp(:,:,nu) = gm(:,:,nu)+gP*Phat(nu)
            A(:,:,nu,1) = glp*Phat(nu)
            A(:,:,nu,2) = -Im*Phat(nu)*id4
            A(:,:,nu,3) = glp*lperp(nu)
            A(:,:,nu,4) = Im*lperp(nu)*id4
            A(:,:,nu,5) = gperp(:,:,nu)-A(:,:,nu,3)
            A(:,:,nu,6) = Im*matmul(gperp(:,:,nu),glp)-A(:,:,nu,4)
        end do
        do n=1,6
            do nu=1,4
                Ba(:,:,nu,n) = matmul(g5,A(:,:,nu,n))
            end do
        end do
    end subroutine

    !dual bases (already transposed): coefficient_m of R = tr2(Ds(:,:,m),R),  sum_nu tr2(Da(:,:,nu,m),R_nu)
    subroutine duals(Bs,Ba,Ds,Da)
        complex(8),intent(in):: Bs(4,4,2),Ba(4,4,4,6)
        complex(8),intent(out):: Ds(4,4,2),Da(4,4,4,6)
        complex(8):: Gs(2,2),Ga(6,6)
        integer:: m,n
        do m=1,2; do n=1,2
            Gs(m,n) = sum(conjg(Bs(:,:,m))*Bs(:,:,n))
        end do; end do
        do m=1,6; do n=1,6
            Ga(m,n) = sum(conjg(Ba(:,:,:,m))*Ba(:,:,:,n))
        end do; end do
        call zinverse(2,Gs); call zinverse(6,Ga)
        Ds = Zero; Da = Zero
        do m=1,2; do n=1,2
            Ds(:,:,m) = Ds(:,:,m)+Gs(m,n)*conjg(transpose(Bs(:,:,n)))
        end do; end do
        do m=1,6; do n=1,6
            Da(:,:,1,m) = Da(:,:,1,m)+Ga(m,n)*conjg(transpose(Ba(:,:,1,n)))
            Da(:,:,2,m) = Da(:,:,2,m)+Ga(m,n)*conjg(transpose(Ba(:,:,2,n)))
            Da(:,:,3,m) = Da(:,:,3,m)+Ga(m,n)*conjg(transpose(Ba(:,:,3,n)))
            Da(:,:,4,m) = Da(:,:,4,m)+Ga(m,n)*conjg(transpose(Ba(:,:,4,n)))
        end do; end do
    end subroutine

end module
