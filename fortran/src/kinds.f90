! Basic kinds and constants.  All momenta in GeV, Euclidean metric a.b = sum_{i=1..4} a_i b_i.
module kinds
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  real(dp),    parameter :: pi = 3.14159265358979323846_dp
  real(dp),    parameter :: hbarc = 0.1973269804_dp          ! GeV fm
  complex(dp), parameter :: zi = (0.0_dp, 1.0_dp), zero = (0.0_dp, 0.0_dp), one = (1.0_dp, 0.0_dp)
contains
  pure complex(dp) function dot4(a, b)
    complex(dp), intent(in) :: a(4), b(4)
    dot4 = sum(a * b)                 ! NB: no complex conjugation (analytic continuation)
  end function dot4
end module kinds
