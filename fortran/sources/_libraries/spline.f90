!Complex cubic spline with 'not-a-knot' end conditions (= scipy CubicSpline default)
module spline
    use constant
    implicit none

public

contains

    !second derivatives m(:,j) for data y(:,j) on knots x (n >= 4), several columns at once
    subroutine spline_setup(n,x,ncol,y,m)
        integer,intent(in):: n,ncol
        real(8),intent(in):: x(n)
        complex(8),intent(in):: y(n,ncol)
        complex(8),intent(out):: m(n,ncol)
        real(8):: a(n,n),h(n-1)
        complex(8):: rhs(n,ncol)
        integer:: i,info,ipiv(n)
        complex(8):: ac(n,n)
        h = x(2:n)-x(1:n-1)
        a = 0d0; rhs = Zero
        do i=2,n-1
            a(i,i-1) = h(i-1); a(i,i) = 2*(h(i-1)+h(i)); a(i,i+1) = h(i)
            rhs(i,:) = 6*((y(i+1,:)-y(i,:))/h(i)-(y(i,:)-y(i-1,:))/h(i-1))
        end do
        !not-a-knot: third derivative continuous at x_2 and x_{n-1}
        a(1,1) = h(2); a(1,2) = -(h(1)+h(2)); a(1,3) = h(1)
        a(n,n-2) = h(n-1); a(n,n-1) = -(h(n-2)+h(n-1)); a(n,n) = h(n-2)
        ac = a
        m = rhs
        call zgesv(n,ncol,ac,n,ipiv,m,n,info)
        if(info/=0) stop 'spline_setup: zgesv failed'
    end subroutine

    pure function spline_eval(n,x,ncol,y,m,t) result(v)
        integer,intent(in):: n,ncol
        real(8),intent(in):: x(n),t
        complex(8),intent(in):: y(n,ncol),m(n,ncol)
        complex(8):: v(ncol)
        integer:: i
        real(8):: h,a,b
        i = 1
        do while(i<n-1 .and. t>x(i+1))
            i = i+1
        end do
        h = x(i+1)-x(i)
        a = (x(i+1)-t)/h; b = (t-x(i))/h
        v = a*y(i,:)+b*y(i+1,:)+((a**3-a)*m(i,:)+(b**3-b)*m(i+1,:))*h*h/6
    end function

end module
