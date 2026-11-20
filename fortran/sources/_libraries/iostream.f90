module iostream
    use constant
    implicit none

public

contains

    subroutine x_mkdir(directory)
        character(len=*):: directory
        call execute_command_line('mkdir -p '//trim(directory))
    end subroutine

end module
