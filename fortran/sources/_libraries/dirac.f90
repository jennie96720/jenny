!Euclidean Dirac algebra (same representation as qdq/dirac.py)
!   gamma_k = [[0,-i sigma_k],[i sigma_k,0]], gamma_4 = diag(1,1,-1,-1)
!   {gamma_mu,gamma_nu} = 2 delta_{mu nu}, gamma_5 = gamma_4 gamma_1 gamma_2 gamma_3, C = gamma_2 gamma_4
!   sigma_{mu nu} = (i/2)[gamma_mu,gamma_nu],  eps_{1234} = +1
module dirac
    use constant
    implicit none

    complex(8):: gm(4,4,4),g5(4,4),cc(4,4),ccdag(4,4),id4(4,4),sig(4,4,4,4)
    real(8):: eps4(4,4,4,4)

public

contains

    subroutine dirac_init()
        complex(8):: s(2,2,3)
        integer:: k,m,n,p,q
        s = Zero
        s(1,2,1) = One;  s(2,1,1) = One
        s(1,2,2) = -Im;  s(2,1,2) = Im
        s(1,1,3) = One;  s(2,2,3) = -One
        gm = Zero
        do k=1,3
            gm(1:2,3:4,k) = -Im*s(:,:,k)
            gm(3:4,1:2,k) =  Im*s(:,:,k)
        end do
        id4 = Zero
        do m=1,4
            id4(m,m) = One
        end do
        gm(:,:,4) = id4;  gm(3,3,4) = -One;  gm(4,4,4) = -One
        g5 = matmul(gm(:,:,4),matmul(gm(:,:,1),matmul(gm(:,:,2),gm(:,:,3))))
        cc = matmul(gm(:,:,2),gm(:,:,4))
        ccdag = conjg(transpose(cc))
        do m=1,4
            do n=1,4
                sig(:,:,m,n) = 0.5d0*Im*(matmul(gm(:,:,m),gm(:,:,n))-matmul(gm(:,:,n),gm(:,:,m)))
            end do
        end do
        eps4 = 0d0
        do m=1,4; do n=1,4; do p=1,4; do q=1,4
            eps4(m,n,p,q) = perm_sign([m,n,p,q])
        end do; end do; end do; end do
    end subroutine

    pure real(8) function perm_sign(ix)
        integer,intent(in):: ix(4)
        integer:: i,j
        perm_sign = 1d0
        do i=1,4
            do j=i+1,4
                if(ix(i)==ix(j))then
                    perm_sign = 0d0; return
                end if
                if(ix(i)>ix(j)) perm_sign = -perm_sign
            end do
        end do
    end function

    pure function slash(p) result(m)
        complex(8),intent(in):: p(4)
        complex(8):: m(4,4)
        integer:: mu
        m = Zero
        do mu=1,4
            m = m + p(mu)*gm(:,:,mu)
        end do
    end function

    !Tr[A B] for 4x4 matrices
    pure complex(8) function tr2(a,b)
        complex(8),intent(in):: a(4,4),b(4,4)
        tr2 = sum(transpose(a)*b)
    end function

    subroutine dirac_check(maxerr)
        real(8),intent(out):: maxerr
        integer:: m,n
        complex(8):: a(4,4)
        maxerr = 0d0
        do m=1,4
            do n=1,4
                a = matmul(gm(:,:,m),gm(:,:,n))+matmul(gm(:,:,n),gm(:,:,m))
                if(m==n) a = a-2*id4
                maxerr = max(maxerr,maxval(abs(a)))
            end do
            a = matmul(cc,matmul(transpose(gm(:,:,m)),ccdag))+gm(:,:,m)    !C gamma^T C^dag = -gamma
            maxerr = max(maxerr,maxval(abs(a)))
        end do
        maxerr = max(maxerr,maxval(abs(matmul(g5,g5)-id4)))
    end subroutine

end module
