! Algebraic dressed light-quark propagator, arXiv:2611.19819 App. A.1 (= qdq/quark.py):
!   S(p) = -i gamma.p sigma_V(p^2) + sigma_S(p^2),  x = p^2/lambda^2,
!   sbar_S(x) = 2 mbar F(2(x+mbar^2)) + F(b1 x) F(b3 x) [b0 + b2 F(eps x)],
!   sbar_V(x) = [1 - F(2(x+mbar^2))]/(x+mbar^2),  F(x) = (1-e^{-x})/x,
!   sigma_S = sbar_S/lambda, sigma_V = sbar_V/lambda^2.
! Entire functions: complex p^2 is allowed (needed inside all bound-state integrals).
module quark
  use kinds
  use dirac
  implicit none
  real(dp), parameter :: lam = 0.566_dp, mbar = 0.00897_dp, b0 = 0.131_dp, b1 = 2.90_dp, &
                         b2 = 0.603_dp, b3 = 0.185_dp, eps = 1.0e-4_dp
contains
  elemental complex(dp) function calF(x)
    complex(dp), intent(in) :: x
    if (abs(x) < 1.0e-6_dp) then
      calF = 1 - x / 2 + x**2 / 6
    else
      calF = (1 - exp(-x)) / x
    end if
  end function calF

  elemental subroutine sigmas(p2, sv, ss)
    complex(dp), intent(in)  :: p2
    complex(dp), intent(out) :: sv, ss
    complex(dp) :: x, xm
    x  = p2 / lam**2
    xm = x + mbar**2
    ss = 2 * mbar * calF(2 * xm) + calF(b1 * x) * calF(b3 * x) * (b0 + b2 * calF(eps * x))
    sv = (1 - calF(2 * xm)) / xm
    sv = sv / lam**2
    ss = ss / lam
  end subroutine sigmas

  elemental complex(dp) function mass_function(p2)
    complex(dp), intent(in) :: p2
    complex(dp) :: sv, ss
    call sigmas(p2, sv, ss)
    mass_function = ss / sv
  end function mass_function

  function propagator(p) result(s)
    complex(dp), intent(in) :: p(4)
    complex(dp) :: s(4,4), sv, ss
    call sigmas(dot4(p, p), sv, ss)
    s = -zi * sv * slash(p) + ss * id4
  end function propagator
end module quark
