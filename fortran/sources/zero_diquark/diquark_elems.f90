!Diquark correlation amplitudes and propagators, arXiv:2611.19819 App. A.2-A.3 (= qdq/diquark.py)
!   Gamma^{0+}(k) = g0 C i gamma_5 F(k^2/w0^2),  Gamma^{1+}_mu(k) = g1 i gamma_mu C F(k^2/w1^2)
!   Gammabar = C^dag Gamma^T C,  Delta^{0+} = F(K^2/w0^2)/m0^2,  Delta^{1+}_{mu nu} = (delta + K K/m1^2) F(K^2/w1^2)/m1^2
!   w^2 = m^2/2.   Flavour matrices t_f, s_f of 2611.19819 (unit norm).
module diquark_elems
    use libraries
    use quark_elems
    implicit none

type diquark_para
    real(8):: m0  = 0.79d0                   !scalar diquark mass
    real(8):: m1  = 0.89d0                   !axial-vector diquark mass
    integer:: nq  = 200                      !radial points (normalisation loop)
    integer:: nz  = 24                       !angular points (normalisation loop)
    real(8):: h   = 1d-3                     !step of d/d alpha
    character(len=40):: root = 'diquark'     !output file outdata/<root>.txt
end type diquark_para

    real(8):: dq_m0 = 0.79d0, dq_m1 = 0.89d0                  !module-wide masses
    complex(8):: gam_s(4,4),bar_s(4,4),gam_a(4,4,4),bar_a(4,4,4)  !Dirac structures (coupling stripped)
    real(8):: tf(2,2,0:3),sf(2,2,0:3)        !flavour matrices (real here)

public

contains

    subroutine diquark_elems_init(m0,m1)
        real(8),intent(in):: m0,m1
        integer:: mu
        complex(8):: tmp(4,4)
        dq_m0 = m0; dq_m1 = m1
        gam_s = matmul(cc,Im*g5)
        bar_s = matmul(ccdag,matmul(transpose(gam_s),cc))
        do mu=1,4
            tmp = Im*matmul(gm(:,:,mu),cc)
            gam_a(:,:,mu) = tmp
            bar_a(:,:,mu) = matmul(ccdag,matmul(transpose(tmp),cc))
        end do
        !t_f^0 = i tau_2/sqrt2 = [[0,1],[-1,0]]/sqrt2, t^1 = (1+tau_3)/2, t^2 = tau_1/sqrt2, t^3 = (1-tau_3)/2
        tf = 0d0
        tf(1,2,0) = 1/sqrt(2d0); tf(2,1,0) = -1/sqrt(2d0)
        tf(1,1,1) = 1d0
        tf(1,2,2) = 1/sqrt(2d0); tf(2,1,2) = 1/sqrt(2d0)
        tf(2,2,3) = 1d0
        !s_f^0 = 1, s^1 = (tau_1 - i tau_2)/sqrt6 = [[0,0],[2,0]]/sqrt6, s^2 = -tau_3/sqrt3, s^3 = -(tau_1 + i tau_2)/sqrt6
        sf = 0d0
        sf(1,1,0) = 1d0; sf(2,2,0) = 1d0
        sf(2,1,1) = 2/sqrt(6d0)
        sf(1,1,2) = -1/sqrt(3d0); sf(2,2,2) = 1/sqrt(3d0)
        sf(1,2,3) = -2/sqrt(6d0)
    end subroutine

    elemental complex(8) function Dprop0(K2)          !Delta^{0+}(K^2)
        complex(8),intent(in):: K2
        Dprop0 = calF(K2/(0.5d0*dq_m0**2))/dq_m0**2
    end function

    elemental complex(8) function Dprop1(K2)          !scalar part of Delta^{1+}
        complex(8),intent(in):: K2
        Dprop1 = calF(K2/(0.5d0*dq_m1**2))/dq_m1**2
    end function

    function Dprop1mat(K) result(d)                   !Delta^{1+}_{mu nu}(K)
        complex(8),intent(in):: K(4)
        complex(8):: d(4,4)
        integer:: m,n
        do m=1,4
            do n=1,4
                d(m,n) = K(m)*K(n)/dq_m1**2
            end do
            d(m,m) = d(m,m)+One
        end do
        d = d*Dprop1(dot4(K,K))
    end function

    !F(k^2/w^2) of the scalar ('s') or axial ('a') amplitude
    elemental complex(8) function Famp(ch,k2)
        character,intent(in):: ch
        complex(8),intent(in):: k2
        if(ch=='s')then
            Famp = calF(k2/(0.5d0*dq_m0**2))
        else
            Famp = calF(k2/(0.5d0*dq_m1**2))
        end if
    end function

end module
