!Algebraic dressed light-quark propagator, arXiv:2611.19819 App. A.1 (= qdq/quark.py)
!   S(p) = -i gamma.p sigma_V(p^2) + sigma_S(p^2),  x = p^2/lambda^2
!   sbar_S(x) = 2 mbar F(2(x+mbar^2)) + F(b1 x) F(b3 x) [b0 + b2 F(eps x)]
!   sbar_V(x) = [1 - F(2(x+mbar^2))]/(x+mbar^2),  sigma_S = sbar_S/lambda, sigma_V = sbar_V/lambda^2
!Entire functions: complex p^2 is allowed.
module quark_elems
    use libraries
    implicit none

type quark_para
    real(8):: lam  = 0.566d0
    real(8):: mbar = 0.00897d0
    real(8):: b0   = 0.131d0
    real(8):: b1   = 2.90d0
    real(8):: b2   = 0.603d0
    real(8):: b3   = 0.185d0
    real(8):: eps  = 1d-4
end type quark_para

    type(quark_para):: qpar          !module-wide parameters (set by quark_para_load)

public

contains

    elemental subroutine sigmas(p2,sv,ss)
        complex(8),intent(in):: p2
        complex(8),intent(out):: sv,ss
        complex(8):: x,xm
        x  = p2/qpar%lam**2
        xm = x+qpar%mbar**2
        ss = 2*qpar%mbar*calF(2*xm)+calF(qpar%b1*x)*calF(qpar%b3*x)*(qpar%b0+qpar%b2*calF(qpar%eps*x))
        sv = (1-calF(2*xm))/xm
        sv = sv/qpar%lam**2
        ss = ss/qpar%lam
    end subroutine

    elemental complex(8) function sigV(p2)
        complex(8),intent(in):: p2
        complex(8):: ss
        call sigmas(p2,sigV,ss)
    end function

    elemental complex(8) function sigS(p2)
        complex(8),intent(in):: p2
        complex(8):: sv
        call sigmas(p2,sv,sigS)
    end function

    elemental complex(8) function mass_function(p2)
        complex(8),intent(in):: p2
        complex(8):: sv,ss
        call sigmas(p2,sv,ss)
        mass_function = ss/sv
    end function

    function propagator(p) result(s)
        complex(8),intent(in):: p(4)
        complex(8):: s(4,4),sv,ss
        call sigmas(dot4(p,p),sv,ss)
        s = -Im*sv*slash(p)+ss*id4
    end function

end module
