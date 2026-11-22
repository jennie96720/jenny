program main
    use libraries
    implicit none
    character(len=20):: mode
    namelist /run_list/ mode

    call MPI_initialize()
    call dirac_init()
    mode = 'check'
    open(unit=100,file='Configures.nlt',status='old')
    read(unit=100,nml=run_list)
    close(unit=100)

    select case(trim(mode))
    case('check');   call run_checks('Configures.nlt')
    case('faddeev'); call run_faddeev('Configures.nlt')
    case('em');      call run_form_factors('Configures.nlt','em')
    case('gff');     call run_form_factors('Configures.nlt','grav')
    case('pdf');     call run_pdf('Configures.nlt',.false.)
    case('helicity');call run_pdf('Configures.nlt',.true.)
    case('axial_table'); call run_axial_table('Configures.nlt')
    case('da');      call run_da('Configures.nlt')
    case default
        if(jrank==0) print*,'unknown mode: ',trim(mode)
    end select

    call MPI_cleanup()
end program
