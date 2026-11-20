module faddeev_utils
    use libraries
    use faddeev_elems
    implicit none

contains

    function faddeev_para_load(fstr) result(para)
        character(len=*):: fstr
        type(faddeev_para):: para
        complex(8):: gs,ga
        namelist /faddeev_para_list/ para
        namelist /diquark_coupling_list/ gs,ga
        open(unit=100,file=trim(fstr),status='old')
        read(unit=100,nml=faddeev_para_list)
        close(unit=100)
        if(para%own_g)then
            open(unit=100,file='outdata/diquark.txt',status='old')
            read(unit=100,nml=diquark_coupling_list)
            close(unit=100)
            para%gs = gs; para%ga = ga
        end if
    end function

end module
