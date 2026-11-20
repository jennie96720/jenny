module diquark_utils
    use libraries
    use diquark_elems
    implicit none

contains

    function diquark_para_load(fstr) result(para)
        character(len=*):: fstr
        type(diquark_para):: para
        namelist /diquark_para_list/ para
        open(unit=100,file=trim(fstr),status='old')
        read(unit=100,nml=diquark_para_list)
        close(unit=100)
    end function

end module
