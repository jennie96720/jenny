!Thin MPI layer (same interface style as the meson code); compiles serially without -DMPI
module mpich2
    use constant
#ifdef MPI
    use mpi
#endif
    implicit none

private

    integer:: jrank=0,jsize=1,mpierror

interface MPI_reducesum
    module procedure MPI_reducesum_Z1
    module procedure MPI_reducesum_Z2
end interface

public jrank,jsize,MPI_initialize,MPI_cleanup,MPI_distribute,MPI_reducesum,MPI_wait_others

contains

subroutine MPI_initialize()
#ifdef MPI
    call MPI_init(mpierror)
    call MPI_comm_rank(MPI_comm_world,jrank,mpierror)
    call MPI_comm_size(MPI_comm_world,jsize,mpierror)
#else
    jrank = 0; jsize = 1
#endif
end subroutine

subroutine MPI_cleanup()
#ifdef MPI
    call MPI_finalize(mpierror)
#endif
end subroutine

subroutine MPI_wait_others()
#ifdef MPI
    call MPI_barrier(MPI_comm_world,mpierror)
#endif
end subroutine

!jobs 1..n are dealt out round-robin: this rank does i = jrank+1, jrank+1+jsize, ...
pure logical function MPI_distribute(i)
    integer,intent(in):: i
    MPI_distribute = (mod(i-1,jsize)==jrank)
end function

subroutine MPI_reducesum_Z1(x)
    complex(8):: x(:)
#ifdef MPI
    complex(8),allocatable:: y(:)
    allocate(y(size(x)))
    call MPI_allreduce(x,y,size(x),MPI_double_complex,MPI_sum,MPI_comm_world,mpierror)
    x = y
#endif
end subroutine

subroutine MPI_reducesum_Z2(x)
    complex(8):: x(:,:)
#ifdef MPI
    complex(8),allocatable:: y(:,:)
    allocate(y(size(x,1),size(x,2)))
    call MPI_allreduce(x,y,size(x),MPI_double_complex,MPI_sum,MPI_comm_world,mpierror)
    x = y
#endif
end subroutine

end module
