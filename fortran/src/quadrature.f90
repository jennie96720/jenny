! Quadrature rules, identical to qdq/quadrature.py, so grid-dependent numbers (e.g. the Faddeev
! eigenvalue on a given grid) can be compared digit by digit with the Python reference.
module quadrature
  use kinds
  implicit none
contains
  subroutine gauleg(n, x, w)                 ! Gauss-Legendre on (-1,1), Newton iteration
    integer, intent(in) :: n
    real(dp), intent(out) :: x(n), w(n)
    integer :: i, j, it
    real(dp) :: z, z1, p1, p2, p3, pp
    do i = 1, (n + 1) / 2
      z = cos(pi * (i - 0.25_dp) / (n + 0.5_dp))
      do it = 1, 100
        p1 = 1; p2 = 0
        do j = 1, n
          p3 = p2; p2 = p1
          p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j
        end do
        pp = n * (z * p1 - p2) / (z * z - 1)
        z1 = z; z = z1 - p1 / pp
        if (abs(z - z1) < 1.0e-15_dp) exit
      end do
      x(i) = -z;  x(n + 1 - i) = z
      w(i) = 2 / ((1 - z * z) * pp * pp);  w(n + 1 - i) = w(i)
    end do
  end subroutine gauleg

  subroutine radial(n, scale, q, w)          ! q = s t/(1-t), t in (0,1) Gauss-Legendre
    integer, intent(in) :: n
    real(dp), intent(in) :: scale
    real(dp), intent(out) :: q(n), w(n)
    real(dp) :: t(n), wt(n)
    call gauleg(n, t, wt)
    t = 0.5_dp * (t + 1);  wt = 0.5_dp * wt
    q = scale * t / (1 - t)
    w = wt * scale / (1 - t)**2
  end subroutine radial

  subroutine chebyshev_u(n, z, w)            ! int_{-1}^{1} dz sqrt(1-z^2) f(z)
    integer, intent(in) :: n
    real(dp), intent(out) :: z(n), w(n)
    integer :: k
    do k = 1, n
      z(k) = cos(k * pi / (n + 1))
      w(k) = pi / (n + 1) * sin(k * pi / (n + 1))**2
    end do
  end subroutine chebyshev_u
end module quadrature
