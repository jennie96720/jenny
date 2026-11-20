! Euclidean Dirac algebra, same representation as qdq/dirac.py:
!   gamma_k = [[0, -i sigma_k], [i sigma_k, 0]],  gamma_4 = diag(1, 1, -1, -1),
!   {gamma_mu, gamma_nu} = 2 delta_{mu nu},  gamma_5 = gamma_4 gamma_1 gamma_2 gamma_3,  C = gamma_2 gamma_4.
! Matrices are complex(dp) (4,4); g(:,:,mu) is gamma_mu.
module dirac
  use kinds
  implicit none
  complex(dp) :: g(4,4,4), g5(4,4), cc(4,4), ccdag(4,4), id4(4,4)
contains
  subroutine dirac_init()
    complex(dp) :: s(2,2,3), z2(2,2)
    integer :: k, m
    z2 = zero
    s = zero
    s(1,2,1) = one;  s(2,1,1) = one                      ! sigma_1
    s(1,2,2) = -zi;  s(2,1,2) = zi                       ! sigma_2
    s(1,1,3) = one;  s(2,2,3) = -one                     ! sigma_3
    g = zero
    do k = 1, 3
      g(1:2,3:4,k) = -zi * s(:,:,k)
      g(3:4,1:2,k) =  zi * s(:,:,k)
    end do
    id4 = zero
    do m = 1, 4
      id4(m,m) = one
    end do
    g(:,:,4) = id4;  g(3,3,4) = -one;  g(4,4,4) = -one
    g5 = matmul(g(:,:,4), matmul(g(:,:,1), matmul(g(:,:,2), g(:,:,3))))
    cc = matmul(g(:,:,2), g(:,:,4))
    ccdag = conjg(transpose(cc))
  end subroutine dirac_init

  pure function slash(p) result(m)
    complex(dp), intent(in) :: p(4)
    complex(dp) :: m(4,4)
    integer :: mu
    m = zero
    do mu = 1, 4
      m = m + p(mu) * g(:,:,mu)
    end do
  end function slash

  subroutine dirac_check(maxerr)
    real(dp), intent(out) :: maxerr
    integer :: m, n
    complex(dp) :: a(4,4)
    maxerr = 0
    do m = 1, 4
      do n = 1, 4
        a = matmul(g(:,:,m), g(:,:,n)) + matmul(g(:,:,n), g(:,:,m))
        if (m == n) a = a - 2 * id4
        maxerr = max(maxerr, maxval(abs(a)))
      end do
      ! charge conjugation: C gamma_mu^T C^dagger = -gamma_mu
      a = matmul(cc, matmul(transpose(g(:,:,m)), ccdag)) + g(:,:,m)
      maxerr = max(maxerr, maxval(abs(a)))
    end do
    maxerr = max(maxerr, maxval(abs(matmul(g5, g5) - id4)))
  end subroutine dirac_check
end module dirac
