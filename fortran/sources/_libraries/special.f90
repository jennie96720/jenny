module special
    use constant
    implicit none

public

contains

    !F(x) = (1-e^{-x})/x
    elemental complex(8) function calF(x)
        complex(8),intent(in):: x
        if(abs(x)<1d-6)then
            calF = 1-x/2+x**2/6
        else
            calF = (1-exp(-x))/x
        end if
    end function

end module
