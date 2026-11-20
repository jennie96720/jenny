module linearsys
    use constant
    implicit none

public

contains

    !eigenvalue with largest real part and its right eigenvector (LAPACK zgeev)
    subroutine eig_maxreal(n,a,ev,vec)
        integer,intent(in):: n
        complex(8),intent(in):: a(n,n)
        complex(8),intent(out):: ev,vec(n)
        complex(8),allocatable:: b(:,:),w(:),vr(:,:),work(:)
        complex(8):: vl(1,1)
        real(8),allocatable:: rwork(:)
        integer:: info,lwork,j
        allocate(b(n,n),w(n),vr(n,n),rwork(2*n))
        b = a
        lwork = -1
        allocate(work(1))
        call zgeev('N','V',n,b,n,w,vl,1,vr,n,work,lwork,rwork,info)
        lwork = int(work(1)); deallocate(work); allocate(work(lwork))
        call zgeev('N','V',n,b,n,w,vl,1,vr,n,work,lwork,rwork,info)
        if(info/=0) stop 'eig_maxreal: zgeev failed'
        j = maxloc(real(w),1)
        ev = w(j); vec = vr(:,j)
    end subroutine

    subroutine zinverse(n,a)
        integer,intent(in):: n
        complex(8),intent(inout):: a(n,n)
        integer:: ipiv(n),info
        complex(8):: work(64*n)
        call zgetrf(n,n,a,n,ipiv,info)
        if(info/=0) stop 'zinverse: singular'
        call zgetri(n,a,n,ipiv,work,64*n,info)
    end subroutine

end module
