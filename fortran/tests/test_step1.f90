! Step 1 check: Dirac algebra, quark propagator values, constituent mass, quadrature nodes.
! Compare with docs/reference_values.md (generated from the Python code).
program test_step1
  use kinds
  use dirac
  use quark
  use quadrature
  implicit none
  real(dp) :: err, a, b, c, q(4), w(4)
  complex(dp) :: p2(5), sv, ss
  integer :: i
  call dirac_init()
  call dirac_check(err)
  print '(a,es10.2)', 'Dirac algebra / charge conjugation max error: ', err
  p2 = [(0.0_dp, 0.0_dp), (0.1_dp, 0.0_dp), (1.0_dp, 0.0_dp), (-0.1_dp, 0.2_dp), (-0.25_dp, 0.5_dp)]
  print '(a)', '   p^2                 sigma_V                      sigma_S                      M(p^2)'
  do i = 1, 5
    call sigmas(p2(i), sv, ss)
    print '(2f7.3,3(2x,2f13.9))', p2(i), sv, ss, mass_function(p2(i))
  end do
  ! Euclidean constituent mass: p = M(p^2) by bisection
  a = 0.05_dp;  b = 2.0_dp
  do i = 1, 100
    c = 0.5_dp * (a + b)
    if (f(a) * f(c) <= 0) then
      b = c
    else
      a = c
    end if
  end do
  print '(a,f14.10,a)', 'M^E = ', c, ' GeV   (reference 0.3275076942)'
  call radial(4, 0.6_dp, q, w)
  print '(a,4f14.10)', 'radial nodes (n=4, s=0.6): ', q
contains
  real(dp) function f(p)
    real(dp), intent(in) :: p
    f = p**2 - real(mass_function(cmplx(p**2, 0.0_dp, kind=dp)))**2
  end function f
end program test_step1
