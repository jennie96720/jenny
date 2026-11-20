module current1_utils
    use libraries
    use current1_elems
    implicit none

contains

    function current_para_load(fstr) result(para)
        character(len=*):: fstr
        type(current_para):: para
        namelist /current_para_list/ para
        open(unit=100,file=trim(fstr),status='old')
        read(unit=100,nml=current_para_list)
        close(unit=100)
    end function

end module
