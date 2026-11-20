module quark_utils
    use libraries
    use quark_elems
    implicit none

contains

    subroutine quark_para_load(fstr)
        character(len=*):: fstr
        type(quark_para):: para
        namelist /quark_para_list/ para
        open(unit=100,file=trim(fstr),status='old')
        read(unit=100,nml=quark_para_list)
        close(unit=100)
        qpar = para
    end subroutine

end module
