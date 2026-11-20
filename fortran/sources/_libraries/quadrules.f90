!Quadrature rules identical to qdq/quadrature.py (so grid-dependent numbers can be compared digit by digit)
module quadrules
    use constant
    implicit none

public

contains

    subroutine gauleg(n,x,w)            !Gauss-Legendre on (-1,1)
        integer,intent(in):: n
        real(8),intent(out):: x(n),w(n)
        integer:: i,j,it
        real(8):: z,z1,p1,p2,p3,pp
        do i=1,(n+1)/2
            z = cos(pi*(i-0.25d0)/(n+0.5d0))
            do it=1,100
                p1 = 1d0; p2 = 0d0
                do j=1,n
                    p3 = p2; p2 = p1
                    p1 = ((2*j-1)*z*p2-(j-1)*p3)/j
                end do
                pp = n*(z*p1-p2)/(z*z-1)
                z1 = z; z = z1-p1/pp
                if(abs(z-z1)<1d-15) exit
            end do
            x(i) = -z;  x(n+1-i) = z
            w(i) = 2/((1-z*z)*pp*pp);  w(n+1-i) = w(i)
        end do
    end subroutine

    subroutine radial(n,scale,q,w)      !q = s t/(1-t), t Gauss-Legendre on (0,1)
        integer,intent(in):: n
        real(8),intent(in):: scale
        real(8),intent(out):: q(n),w(n)
        real(8):: t(n),wt(n)
        call gauleg(n,t,wt)
        t = 0.5d0*(t+1);  wt = 0.5d0*wt
        q = scale*t/(1-t)
        w = wt*scale/(1-t)**2
    end subroutine

    subroutine chebyshev_u(n,z,w)       !int_{-1}^{1} dz sqrt(1-z^2) f(z)
        integer,intent(in):: n
        real(8),intent(out):: z(n),w(n)
        integer:: k
        do k=1,n
            z(k) = cos(k*pi/(n+1))
            w(k) = pi/(n+1)*sin(k*pi/(n+1))**2
        end do
    end subroutine

    !Chebyshev polynomials of the 2nd kind U_0..U_{n-1} at (complex) z
    pure function cheb_u(n,z) result(u)
        integer,intent(in):: n
        complex(8),intent(in):: z
        complex(8):: u(n)
        integer:: k
        u(1) = One
        if(n>1) u(2) = 2*z
        do k=3,n
            u(k) = 2*z*u(k-1)-u(k-2)
        end do
    end function

end module
