module constant
    implicit none

    real(8),parameter         ::  pi     = 3.141592653589793238462d0
    real(8),parameter         ::  hbarc  = 0.1973269804d0      !GeV fm
    complex(8),parameter      ::  Im     = (0.d0,1.d0)
    complex(8),parameter      ::  Zero   = (0.d0,0.d0)
    complex(8),parameter      ::  One    = (1.d0,0.d0)

public

contains

    !Euclidean scalar product a.b = sum_i a_i b_i  (no complex conjugation: analytic continuation)
    pure complex(8) function dot4(a,b)
        complex(8),intent(in):: a(4),b(4)
        dot4 = sum(a*b)
    end function

end module
